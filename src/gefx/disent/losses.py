"""As perdas da fase 1: contrastivo supervisionado, reversao de gradiente,
ortogonalidade -- e o registro que as compoe.

O quadrante da taxonomia de Wang et al. em que este trabalho cai e
*vector-wise, supervisionado, plano, com independencia*: a grade totalmente
cruzada rotula todos os fatores, entao nao ha por que inferi-los. Cada peca
daqui responde por um pedaco desse quadrante:

- **contrastivo supervisionado** (Khosla et al. 2020) sobre `z_e`, com a
  configuracao como classe: puxa junto o que compartilha ajuste ainda que mude
  conteudo e implementacao. E a unica perda que fala diretamente da metrica de
  recuperacao, porque a busca e por cosseno no mesmo `z_e`.
- **reversao de gradiente** (Ganin & Lempitsky 2015): um classificador de
  implementacao e outro de conteudo penduram em `z_e`; a camada de reversao faz
  o encoder *piorar* os dois. A rampa de lambda e a do artigo -- adversario forte
  desde o passo zero derruba o treino antes de haver o que remover.
- **ortogonalidade**: descorrelaciona `z_e` de `z_c` no batch. Sem ela nada
  impede que a mesma informacao viva nos dois, e a divisao vira decorativa.

O registro `LOSS_REGISTRY` e a terceira costura do compromisso da fase 2 (ver a
memoria `poc2-extensao-decoder-troca-de-codigos`): a perda do laco de treino e a
soma ponderada das entradas de um `dict[str, callable]`, entao `swap_recon` entra
como mais uma chave, sem tocar no laco.

Toda funcao recebe o mesmo contexto -- um dicionario com os codigos, os logits
das cabecas e os rotulos do batch. Assinatura unica de proposito: e o que permite
acrescentar uma perda sem mexer em quem chama.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional

EPSILON = 1e-8

#: Temperatura do contrastivo. 0,07 e o valor do SimCLR/SupCon; com `z_e`
#: L2-normalizado o produto interno vive em [-1, 1] e a temperatura e o unico
#: controle da dureza dos negativos.
DEFAULT_TEMPERATURE = 0.07

#: Ganho da rampa de lambda. gamma=10 e o do Ganin.
RAMP_GAMMA = 10.0


# --- reversao a gradiente -----------------------------------------------------
#: `tf.custom_gradient` so pode ser aplicado com o TensorFlow ja importado, e
#: este pacote importa TF **dentro** das funcoes (o teste de higiene de imports
#: cobra isso: e o que mantem a suite em segundos). Dai a construcao preguicosa,
#: feita uma vez e guardada.
_REVERSE: Optional[Any] = None


def _reverse_op():
    global _REVERSE
    if _REVERSE is None:
        import tensorflow as tf

        @tf.custom_gradient
        def reverse(x, lam):
            def grad(upstream):
                # Nenhum gradiente volta para `lam`: ele e agenda, nao parametro.
                return -lam * upstream, None

            return tf.identity(x), grad

        _REVERSE = reverse
    return _REVERSE


def gradient_reversal(x, lam):
    """Identidade na ida, gradiente multiplicado por `-lam` na volta.

    O adversario ve o codigo intacto e aprende normalmente; o encoder recebe o
    gradiente com o sinal trocado e aprende a *atrapalhar* o adversario. As duas
    coisas num passo so, que e a razao de o truque existir.
    """
    import tensorflow as tf

    return _reverse_op()(x, tf.cast(lam, x.dtype))


def lambda_ramp(progress: float, gamma: float = RAMP_GAMMA) -> float:
    """Rampa `2 / (1 + exp(-gamma * p)) - 1`, com `p` em [0, 1] (Ganin & Lempitsky).

    Sai de 0 e satura em 1. O comeco suave e o que importa: com o adversario a
    todo peso desde o inicio, o encoder aprende a apagar `z_e` inteiro -- e um
    codigo constante engana qualquer classificador de implementacao.
    """
    p = float(min(max(progress, 0.0), 1.0))
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# --- contrastivo supervisionado ----------------------------------------------
def supervised_contrastive(z, labels, temperature: float = DEFAULT_TEMPERATURE):
    """L_out^sup de Khosla et al. (2020), com `z` ja L2-normalizado.

    Para cada ancora, os positivos sao **todas** as outras entradas do batch com
    a mesma configuracao -- e nao um unico par, como no InfoNCE. E por isso que o
    batch precisa ser balanceado por classe: uma ancora sem positivo nao produz
    termo nenhum, e um batch sorteado uniformemente sobre 40 configuracoes quase
    nao tem positivos.
    """
    import tensorflow as tf

    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    batch = tf.shape(z)[0]

    logits = tf.matmul(z, z, transpose_b=True) / temperature
    # Estabilidade numerica: subtrair o maximo da linha nao muda o softmax.
    logits = logits - tf.stop_gradient(tf.reduce_max(logits, axis=1, keepdims=True))

    not_self = 1.0 - tf.eye(batch, dtype=logits.dtype)
    same = tf.cast(tf.equal(labels[:, None], labels[None, :]), logits.dtype)
    positives = same * not_self

    log_denominator = tf.math.log(
        tf.reduce_sum(tf.exp(logits) * not_self, axis=1) + EPSILON
    )
    log_probability = logits - log_denominator[:, None]

    n_positives = tf.reduce_sum(positives, axis=1)
    per_anchor = tf.reduce_sum(positives * log_probability, axis=1) / tf.maximum(
        n_positives, 1.0
    )
    # Ancoras sem positivo nao entram na media: incluir zeros diluiria a perda em
    # funcao da composicao do batch, e nao do que a rede aprendeu.
    valid = tf.cast(n_positives > 0.0, logits.dtype)
    return -tf.reduce_sum(per_anchor * valid) / tf.maximum(tf.reduce_sum(valid), 1.0)


# --- independencia entre os dois codigos --------------------------------------
def orthogonality(z_e, z_c):
    """Correlacao cruzada media ao quadrado entre `z_e` e `z_c` no batch.

    Padroniza cada dimensao antes de correlacionar, entao o valor nao depende da
    escala dos codigos e fica em [0, 1]: 0 e independencia linear completa, 1 e
    duplicacao. Isso torna o peso desta perda comparavel entre execucoes, o que
    uma norma de Frobenius crua nao seria.

    E independencia **linear**, nao independencia estatistica -- limite honesto
    da penalidade, e a razao de as sondas da etapa 6 existirem.
    """
    import tensorflow as tf

    z_e = tf.cast(z_e, tf.float32)
    z_c = tf.cast(z_c, tf.float32)
    n = tf.cast(tf.shape(z_e)[0], tf.float32)

    def standardize(z):
        mean, variance = tf.nn.moments(z, axes=[0], keepdims=True)
        return (z - mean) / tf.sqrt(variance + EPSILON)

    cross = tf.matmul(standardize(z_e), standardize(z_c), transpose_a=True) / n
    return tf.reduce_mean(tf.square(cross))


# --- o registro ---------------------------------------------------------------
LossFn = Callable[[Mapping[str, Any]], Any]


def _cross_entropy(logits, labels):
    import tensorflow as tf

    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(tf.reshape(labels, [-1]), tf.int32), logits=logits
        )
    )


def loss_contrastive(ctx: Mapping[str, Any]):
    return supervised_contrastive(
        ctx["z_e"], ctx["config_label"], ctx.get("temperature", DEFAULT_TEMPERATURE)
    )


def loss_aux_regression(ctx: Mapping[str, Any]):
    """Regressao auxiliar dos niveis normalizados, direto de `z_e`.

    Nao substitui o contrastivo: ancora a *ordem* dos eixos, que o contrastivo
    sozinho nao ve -- para ele as 40 configuracoes sao 40 classes sem vizinhanca,
    e errar por um nivel custa o mesmo que errar por sete.
    """
    import tensorflow as tf

    return tf.reduce_mean(tf.square(ctx["aux_prediction"] - ctx["aux_target"]))


def loss_adversary_arm(ctx: Mapping[str, Any]):
    return _cross_entropy(ctx["adv_arm_logits"], ctx["arm_label"])


def loss_adversary_content(ctx: Mapping[str, Any]):
    return _cross_entropy(ctx["adv_content_logits"], ctx["content_label"])


def loss_adversary_config(ctx: Mapping[str, Any]):
    """Adversario de configuracao em `z_c`: o simetrico dos outros dois.

    Sem ele a separacao seria unilateral -- `z_e` limpo de conteudo, mas `z_c`
    livre para guardar tambem a configuracao, e ai a divisao nao e uma divisao.
    """
    return _cross_entropy(ctx["adv_config_logits"], ctx["config_label"])


def loss_orthogonality(ctx: Mapping[str, Any]):
    return orthogonality(ctx["z_e"], ctx["z_c"])


#: Cada chave e um termo da perda total. Acrescentar `swap_recon` na fase 2 e
#: acrescentar uma entrada aqui e um peso na configuracao -- nada mais.
LOSS_REGISTRY: Dict[str, LossFn] = {
    "contrastive": loss_contrastive,
    "aux_regression": loss_aux_regression,
    "adversary_arm": loss_adversary_arm,
    "adversary_content": loss_adversary_content,
    "adversary_config": loss_adversary_config,
    "orthogonality": loss_orthogonality,
}


def total_loss(
    ctx: Mapping[str, Any], weights: Mapping[str, float]
) -> Dict[str, Any]:
    """Soma ponderada dos termos pedidos, mais cada termo cru para o historico.

    Peso zero **remove** o termo em vez de multiplica-lo por zero: e assim que se
    monta o estudo comparativo de tecnicas (so contrastivo, contrastivo + GRL,
    etc.) sem ramo condicional no laco de treino, e o que nao entra tambem nao
    gasta computo.
    """
    import tensorflow as tf

    unknown = set(weights) - set(LOSS_REGISTRY)
    if unknown:
        raise KeyError(
            f"perdas desconhecidas: {sorted(unknown)}. "
            f"Registradas: {sorted(LOSS_REGISTRY)}"
        )
    parts: Dict[str, Any] = {}
    total = tf.constant(0.0, dtype=tf.float32)
    for name, weight in weights.items():
        if weight == 0.0:
            continue
        value = tf.cast(LOSS_REGISTRY[name](ctx), tf.float32)
        parts[name] = value
        total = total + weight * value
    parts["total"] = total
    return parts

Escuta cega para medir o vies perceptual residual da grade do POC II.

Feita depois de a calibracao (descritor combinado THD x planicidade, level_mode
"descriptor") reportar desvio maximo de 0,25 dB em todos os arms -- ou seja, com
a grade ja alinhada pelo descritor. O objetivo era saber se o alinhamento medido
corresponde ao percebido.

PROCEDIMENTO
Cada ensaio e um wav unico: a REFERENCIA (pedalboard-tanh) no nivel p, pausa de
0,7 s, e depois candidatos do MESMO arm em niveis vizinhos, com pausas de 0,5 s,
em ordem sorteada. O avaliador escolhe qual candidato casa melhor a referencia.
Conteudo, nivel e arm sao os mesmos dentro de um ensaio; so o nivel do candidato
varia. O avaliador nao via o gabarito.

  rodada 1  18 ensaios, 3 alternativas (p-1, p, p+1)
            arms: byod-bigmuff, lsp-hardclip, lsp-tanh (controle)
            niveis 2, 4, 6; 2 conteudos
  rodada 2   8 ensaios, 4 alternativas (p-2, p-1, p, p+1)
            arms: byod-bigmuff (6), lsp-tanh (2, controle)
            niveis 2, 4, 6; 2 conteudos

  rodada 3   6 ensaios, 4 alternativas -- lsp-hardclip
  rodada 4   9 ensaios, 4 alternativas -- lsp-hardclip, niveis 4/5/6, conteudos
             novos. Concentrada no lado sujo de proposito: la o julgamento
             existe. VIESA a estimativa para a metade suja do eixo.
  grade A/B  12 ensaios, 2 alternativas -- mesmo arm e mesmo nivel vindos das
             duas condicoes da grade (niveis casados x uniformes no knob)
  tom        2 ordenacoes de 5 itens -- niveis de tom embaralhados, em drive
             baixo (d1) e alto (d7)

RESULTADOS
  byod-bigmuff   vies -1,17 niveis, IC95 [-1,96; -0,38], n=12 (soa MAIS sujo)
                 direcao sem excecao: p+1 nunca foi escolhido
  lsp-hardclip   vies +0,78, IC95 [+0,44; +1,12], n=9 julgaveis (soa MAIS limpo)
                 SINAL OPOSTO ao do bigmuff -- o erro do descritor e sinalizado
  lsp-tanh       controle, 6/8 acertos; os 2 erros no nivel mais limpo
  grade A/B      10/12 preferiram a grade casada (p=0,019), e a preferencia
                 ACOMPANHA o tamanho do efeito: 3/3 no hard clip (11,7 dB de
                 diferenca), 3/3 no bigmuff (7,4), 3/3 no mxr (4,6), 1/3 no sine
                 (3,7). rho preferencia x efeito = +0,60 (p=0,039). A relacao
                 dose-resposta e o que descarta vies de resposta.
  tom            5/5 exatos nos DOIS arquivos, rho +1,000. O controle positivo
                 sobrevive ao drive alto: a distorcao pesada nao mascara o filtro.

O ORACULO ACERTOU OS DOIS SINAIS
  bigmuff   ouvido -1,17   oraculo -1,76   descritor da calibracao: "casado"
  hardclip  ouvido +0,78   oraculo +1,16   descritor da calibracao: "casado"
Planicidade, crest, centroide e o descritor combinado falham em pelo menos um.

O QUE O AVALIADOR RELATOU, E QUE EXPLICA O LIMITE
  "a referencia nao tem chiado, mas todos os outros tem" -- a referencia e um
  clipper SUAVE (tanh); o hard clip trunca e gera harmonicos de ordem alta. Eles
  diferem em ESPECIE, nao em grau, e a pergunta "qual tem mais distorcao"
  pressupoe uma escala comum que ali nao existe. Metade dos ensaios do hard clip
  foram chutes declarados por isso. Nao e falha do teste nem do avaliador: e o
  mesmo limite que derrubou todos os descritores escalares.

LIMITACOES
Um unico avaliador. Amostra pequena. O proprio avaliador relatou que no pe do
eixo as alternativas ficam dificeis de distinguir, e e la que os erros do
controle caem -- o julgamento e menos confiavel com pouco ganho.

O vies esta gravado em `PERCEPTUAL_BIAS`, em src/gefx/disent/arms.py, e vai para
o JSON de calibracao. NAO foi aplicado como correcao dos knobs.

================================================================================
TESTE 3 -- z_e CONTRA z_c (2026-09-10, 24 ensaios, duas perguntas cada)

O primeiro a julgar a REPRESENTACAO e nao o placar. Cada ensaio: uma referencia
e dois candidatos, um escolhido pelo bloco z_e e outro pelo z_c, ambos do MESMO
arm (diferente do da referencia) para cancelar a diferenca de especie de
clipping que limitou as rodadas 3 e 4. Nao usa rotulo nenhum na formulacao.

  P1 distorcao  qual candidato esta mais perto na quantidade de distorcao
  P2 frase      qual candidato esta mais perto na frase tocada

Previsao do desemaranhamento: P1 -> candidato do z_e, P2 -> candidato do z_c, e
portanto respostas DIFERENTES no mesmo ensaio (cruzamento).

RESULTADO
  P2 frase      16/24 = 67% no candidato do z_c, p = 0,152. NAO e vies de
                posicao (71% com o previsto em 1, 60% com o previsto em 2).
                No subconjunto em que z_c recuperou literalmente a mesma frase
                (10 ensaios, condicao gravada no gabarito antes das respostas):
                8/10. Nos outros 14, em que nenhum candidato e a mesma frase e
                nao ha o que acertar: 8/14.
  P1 distorcao  15/24 bruto, mas 50% / 71% conforme a posicao -- o agregado e em
                boa parte preferencia por "2".
  cruzamento    11/24, no acaso.

POR QUE P1 NAO E INTERPRETAVEL
Tres arbitros discordam entre si: ouvido x rotulo nominal 10/20; ouvido x
oraculo 11/22; z_e x oraculo 7/22. Sem verdade fundamental estavel, o nulo nao
distingue "o z_e falhou" de "a pergunta nao tinha resposta".

A CAUSA E UM ERRO DE DESENHO, E ELE E CORRIGIVEL
Os ensaios foram filtrados exigindo >= 2 niveis NOMINAIS de separacao entre os
candidatos -- exatamente a grandeza que este trabalho ja mostrou nao atravessar
implementacoes. O oraculo move o nivel em 22 dos 24 ensaios, em media 1,67
niveis e ate 4. A separacao audivel que o filtro pretendia garantir foi comida
pelo descasamento de rotulo entre arms.

  CONSERTO PARA UMA VERSAO 2: filtrar por separacao em niveis EQUIVALENTES do
  oraculo (diagnostics.oracle_alignment), nao em niveis nominais. Alternativa
  mais limpa para isolar so a afirmacao "z_e carrega quantidade de distorcao":
  referencia e candidatos no MESMO arm, onde o nivel nominal e comparavel.

O QUE O TESTE ENTREGA
O ouvido ordenou as duas metades da afirmacao na MESMA ordem que o estimador de
massa por bloco: o conteudo sai de z_e de forma clara (0,204 contra nula ~0,42)
e a configuracao entra em z_e com pouca folga (0,438 contra 0,363). Dois
instrumentos independentes concordando sobre qual metade e forte -- nao e
confirmacao da tese, e concordancia sobre onde ela e fraca.

LIMITACOES
Um unico avaliador, 24 ensaios, nenhum resultado significativo a 5%.

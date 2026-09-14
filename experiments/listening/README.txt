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

================================================================================
TESTE 4 -- A TROCA DE CODIGOS DA FASE 2 E AUDIVEL?          (montado 2026-09-11)
================================================================================

AFIRMACAO TESTADA, e so ela: a troca de codigos moveu o efeito NA DIRECAO DO
DOADOR E PARA LONGE DA ANCORA. E a metade da afirmacao da fase 2 que o erro
quadratico nao verifica, e a que tem os numeros mais fortes (le a configuracao
do doador a 4,2x o acaso; le a da ancora NO acaso).

O ENSAIO
  referencia  = a gravacao DOADORA        (frase B, config Y, arm_d)
  candidato S = o que a troca recuperou   (frase A, config Y_hat, arm_a)
  candidato N = a propria ancora          (frase A, config X,     arm_a)
  pergunta: qual candidato tem a distorcao mais proxima da referencia?
  acaso: 50%.

POR QUE ESTE DESENHO E MAIS LIMPO QUE OS TRES ANTERIORES
Os dois candidatos sao, por construcao, a MESMA FRASE no MESMO PLUGIN -- ambos
sao x[conteudo(a), *, arm(a)]. Entao (i) o conteudo deixa de ser pista e (ii) o
nivel nominal entre eles e comparavel, e a armadilha que invalidou metade do
teste 3 nao alcanca a comparacao que importa.

FILTROS PRE-DECLARADOS (859 elegiveis de 24.000 pares)
  F1  Y_hat != X: os candidatos tem de ser arquivos diferentes.       7793/8000
  F2  |drive(Y_hat) - drive(X)| >= 2, no mesmo arm (nominal vale).    4914/8000
  F3  doador mapeado para arm_a PELO ORACULO a >= 2 de drive(X).      5389/8000
  F4  tone(Y_hat) == tone(X): candidatos diferem so em drive.         1649/8000
  F5  ambos na metade suja (drive >= 3), onde um degrau e audivel.    3902/8000
Nenhum pergunta se o modelo acertou -- so se o ensaio e respondivel.

O VIES QUE OS FILTROS TINHAM, E O CONSERTO
A primeira montagem saiu com o oraculo dando razao a troca em 24 de 24. Nao era
sorte: o eixo e limitado (0-7), o F2 poe a recuperada longe da ancora, o F3 poe o
doador longe da ancora, e numa escala limitada "os dois longe" quase sempre quer
dizer "os dois do mesmo lado" -- logo perto um do outro. Os filtros escritos para
garantir boa-postura estavam selecionando os casos em que o MODELO ACERTOU, e o
teste teria deixado de medir o modelo: um nulo nao distinguiria "o transplante
nao e audivel" de "o ouvido discorda do modelo".

  CONSERTO: estratificar. 12 ensaios em que o oraculo da razao a troca e 12 em
  que da razao a ancora (entre 859 elegiveis, so ~6% sao discordantes, dai os
  24.000 pares). No estrato discordante o ouvido PRECISA discordar do candidato
  da troca se estiver seguindo o som.

  E o mesmo tipo de erro do teste 3 -- um filtro que parecia garantir boa-postura
  e que selecionava o desfecho -- mas pego ANTES de gastar o tempo do avaliador.

O QUE CADA RESPOSTA VAI RESPONDER
  primaria    o ouvido concorda com o oraculo? Se sim nos DOIS estratos, o
              transplante espectral tem correspondente perceptual e o
              instrumento e valido.
  secundaria  o ouvido segue o som ou o modelo? Escolher a troca nos 24 NAO
              seria confirmacao da fase 2 -- seria sinal de pista que nao e som.

CHECAGENS DE MONTAGEM
  estratos            12 / 12
  posicao x estrato   6/6 e 5/7 -- a posicao sorteada nao se correlaciona com o
                      estrato, entao nao ha pista ai
  LUFS                -26,00 nos 72 arquivos, desvio 0,0001
  referencia no mesmo arm dos candidatos: 3 de 24 (gravado no gabarito)

ARQUIVOS
  datasets/teste_troca/ensaioNN/{referencia,1,2}.wav + LEIA.txt + respostas.csv
  experiments/listening/teste_troca_gabarito.csv
  experiments/listening/build_teste_troca.py   (montagem reproduzivel)

RESPOSTAS (2026-09-13, 24/24, 4 marcados como chute ou quase chute)

UM ERRO DE MONTAGEM MEU, ACHADO NA PONTUACAO
O estrato "oraculo da razao a ancora" foi montado com desigualdade ESTRITA
(dist_troca < dist_ancora -> troca; senao -> ancora). Os EMPATES caram no
estrato da ancora. Dos 12 ensaios desse estrato, 7 sao empates: o doador fica
exatamente no meio dos dois candidatos (ancora 3 ou 7, recuperada 7 ou 3,
doador mapeado 5). Nesses nao ha resposta certa. Sobram 17 bem-postos: 12 em que
o oraculo da razao a troca e 5 em que da razao a ancora.

O avaliador descreveu dois desses empates sem saber que eram empates (ensaios 6
e 9): "achei uma das opcoes com menos drive, a outra com mais". E a descricao
literal de um doador no meio.

  ouvido x oraculo, 24 brutos                   18/24 = 75%   p = 0,023
  ouvido x oraculo, SEM EMPATES                 15/17 = 88%   p = 0,002
    oraculo = troca                             10/12 = 83%   p = 0,039
    oraculo = ancora (o MODELO ERROU)            5/5  = 100%  p = 0,063
  sem empates e sem chutes                      13/15 = 87%   p = 0,007
  escolheu o candidato da troca, 24 brutos      14/24 = 58%   p = 0,54

LEITURA
1. O transplante espectral e AUDIVEL. Quando o oraculo diz que um candidato esta
   mais perto do doador, o ouvido escolhe esse candidato 15 de 17 vezes.
2. O ouvido segue o SOM, e nao o modelo. Nos 5 ensaios em que o modelo errou, o
   ouvido discordou do modelo 5 de 5. E exatamente o controle que o estrato
   existia para oferecer -- com n = 5, p = 0,06 isolado, entao vale como
   coerencia e nao como prova.
3. "Escolheu a troca" em 58% nao e o numero da fase 2: ele mistura acertos,
   erros e empates do modelo, e fica perto de 50% por construcao do desenho.

CHUTES DECLARADOS
  6, 9    empates -- nao havia resposta, e o avaliador disse isso.
  17, 23  doador no nivel 0 do arm da ancora, candidatos em 3 e 5: "ambas com
          drive demais". Nos dois o avaliador escolheu o mais limpo, que e o
          que o oraculo manda. Percepcao certa, declarada como incerta.

POSICAO
Respondeu "2" em 15/24. Nos 7 empates, escolheu "2" em 6 -- a preferencia por
"2" aparece onde nao ha som para decidir, e some onde ha: acerto x oraculo
81,8% com a troca em 1 e 69,2% com a troca em 2, sem empates ainda maior.

LIMITACOES
Um avaliador; 17 ensaios bem-postos; o estrato de erro do modelo tem 5.

LICAO DE MONTAGEM (vai para a memoria)
Estratificar por veredito de um arbitro exige tres classes, nao duas: a favor,
contra e EMPATE. Desigualdade estrita manda o empate calado para um dos lados.

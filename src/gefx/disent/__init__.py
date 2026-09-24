"""POC II: um codigo de efeito desemaranhado para recuperar a configuracao de distorcao.

O POC I regredia o vetor de parametros e o estudo preliminar (`gefx.crossimpl`)
mostrou que isso nao transfere entre implementacoes. Aqui a tarefa e outra:
codificar o audio em `z_efeito` -- informativo sobre a configuracao, invariante a
conteudo e implementacao --, montar um catalogo por implementacao e responder
por busca, devolvendo a configuracao ja nos controles da implementacao consultada.

O pacote nao reaproveita `effects/catalog.py` de proposito -- aquele e a fonte
unica de verdade do POC I e os resultados versionados dependem dele.
"""

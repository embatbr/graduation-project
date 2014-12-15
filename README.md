#TG


##TODO

Melhorias no código:

- Retirar o parâmetro "Transpose" dos métodos de leitura do módulo "corpus". Retornar os MFCCs sempre como uma matriz de ordem numframes x numcoeficientes.

Verificar as seguintes ferramentas que podem melhorar a performance:

- [Numba](https://github.com/numba/numba)
- [MKL Optimizations](https://store.continuum.io/cshop/mkl-optimizations/)
- [Question no stakoverflow](http://stackoverflow.com/questions/16178471/numpy-running-at-half-the-speed-of-matlab)
- [Exemplo de GMM](http://nbviewer.ipython.org/github/tritemio/notebooks/blob/master/Mixture_Model_Fitting.ipynb)


##Proposta

Refazer o projeto da disciplina *Processamento de Voz*, utilizando o artigo de
Reynolds [1] como base. Como adicional devo **tentar** reproduzir o artigo sobre
*fractional covariance* [2] nos GMMs da proposta básica.

Na proposta deve ficar claro que o projeto em si é a reprodução do artigo de
Reynolds [1]. A parte adicional ainda não é possível saber se funcionará para
processamento de voz, pois nunca foi tentada (o artigo utiliza para imagem).

###Roteiro

- Entender a(s) base(s) utilizada(s);
- Implementar VAD(s) (não utilizado em *PV*);
- Extrair os MFCCs;
- Gerar o GMM-UBM;
- Gerar os GMMs para os locutores através do método de adaptação;
- Efetuar os testes (que devem apresentar resultados compatíveis com [1])
- Adicionar as técnicas de *fractional covariance*, repetir os experimentos e
estudar os resultados.

###Trabalhos Futuros

- Escrever papers para conferências e periódicos sobre FPCA em processamento de
voz. Dentre estes, está na mira o INTERSPEECH 2015.


##Entregas

**(OK) Proposta :** Até o dia 29-10-2014, para o professor Ruy Guerra via email
(ruy@cin.ufpe.br) e em papel com a assinatura do orientador (Tsang).

**Relatório :** Ao final do período. Deve ser o mais sucinto possível.


##Referências

+ [1] Speaker Verification Using Adapted Gaussian Mixture Models
+ [2] Theory of fractional covariance matrix and its applications in PCA and 2D-PCA
+ [3] An Algorithm for Determining the Endpoints of Isolated Utterances
+ [4] ROBUST ENDPOINT DETECTION FOR SPEECH RECOGNITION BASED ON DISCRIMINATIVE FEATURE EXTRACTION
+ [5] Voice Activity Detection Based on the Bispectrum
+ [6] A Tutorial on Principal Component Analysis
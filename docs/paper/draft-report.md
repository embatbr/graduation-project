# Draft do TG

## Chapter: Feature Extraction

Falar da importância da extração de características.

### MFCCs

Explicar o que são os MFCCs, o que é a escala MEL, o Cepstrum e etc.

### MFCCs Extraction

Explicar os passos da extração dos MFCCs.

1. Mostrar o sinal de voz original, representado pela imagem "*sigproc/figure000.png*", com **magnitude** e espectro de **potência** dados pelas imagens "*sigproc/figure001.png*" e "*sigproc/figure002.png*". O espectro mostra que o sinal é formado basicamente por frequências baixas, o que complica sua análise.

2. Mostrar imagem do sinal pré-enfatizado com coeficiente 0.97 (vai de 0 a 1), cuja imagem é "*sigproc/figure003.png*", com **magnitude** e espectro de **potência** dados pelas imagens "*sigproc/figure004.png*" e "*sigproc/figure005.png*". As imagens do espectro mostram que o processo de pré-ênfase aumenta a intensidade das frequências mais altas e diminui a das frequências mais baixas (fica mais "flat"). Assim fica mais fácil extrair as características da parte alta (as da parte baixa continuam "extraíveis"). Mostrar o filtro de pré-ênfase no domínio da frequência (TODO fazer as contas).

3. A seguir é explicado o janelamento, utilizando a **Janela de Hamming** (mostrar sua equação e suas imagens nos domínios do tempo e da frequência). Cada frame é dado pela imagem "*sigproc/figure{006,009,...}.png*" . Cada frame também possui uma imagem de **magnitude** e uma de espectro de **potência**, dadas pelos arquivos "*sigproc/figure{007,010,...}.png*" e "*sigproc/figure{008,011,...}.png*". Existem imagens para 10 frames igualmente espaçados ao longo de todo o sinal.

4. Mostrar a escala Mel, imagens "*features/figure{002,003}.png*", e o banco de filtros em escala MEL, representado pela imagem "*features/figure004.png*". Embora seja em escala MEL, a imagem mostra os filtros na escala Hertz. Na escala MEL todos os filtros tem o mesmo comprimento (que na escala Hertz crescem com a frequência). Explicar sobre o funcionamento da Cóclea no ouvido humano e como este banco de filtros simula sua funcionalidade (talvez colocar isso na seção anterior).

5. Mostrar como o espectro de **potência** do sinal com pré-ênfase (imagem "*sigproc/figure005.png*") fica após passar pelo banco de filtros #20 (imagem "*features/figure005.png*"). O resultado é a imagem "*features/figure006.png*". Fazer o mesmo para todo o filtro (imagem "*features/figure004.png*"), com resultado na imagem "*features/figure007.png*" (parecida com a imagem do espectro de **potência** original).
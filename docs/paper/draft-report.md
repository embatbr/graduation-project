# Draft do TG

## Chapter: Feature Extraction

Falar da importância da extração de características.

### MFCCs

Explicar o que são os MFCCs, o que é a escala MEL, o Cepstrum e etc.

### MFCCs Extraction

Explicar os passos da extração dos MFCCs.

1. Mostrar o sinal de voz original, representado pela imagem "*sigproc/part0-signal-enroll_2-f08-54-16000Hz.png*", com **magnitude** e espectro de **potência** dados pelas imagens "*sigproc/part0-signal-enroll_2-f08-54-16000Hz-magspec.png*" e "*sigproc/part0-signal-enroll_2-f08-54-16000Hz-powspec.png*". O espectro mostra que o sinal é formado basicamente por frequências baixas, o que complica sua análise.

2. Mostrar imagem do sinal pré-enfatizado com coeficiente 0.97 (vai de 0 a 1), cuja imagem é "*sigproc/part1-signal-enroll_2-f08-54-16000Hz-preemph0.97.png*", com **magnitude** e espectro de **potência** dados pelas imagens "*sigproc/part1-signal-enroll_2-f08-54-16000Hz-preemph0.97-magspec.png*" e "*sigproc/part1-signal-enroll_2-f08-54-16000Hz-preemph0.97-powspec.png*". As imagens do espectro mostram como o processo de pré-ênfase melhora a visualização das frequências mais altas e mantém as características do sinal.

3. A seguir é explicado o janelamento, utilizando a **Janela de Hamming** (mostrar sua equação e suas imagens nos domínios do tempo e da frequência). Cada frame é dado pela imagem "*sigproc/part2-signal-enroll_2-f08-54-16000Hz-preemph0.97-hamming#.png*" (onde # são os dígitos correspondentes ao índice do frame). Cada frame também possui uma imagem de **magnitude** e uma de espectro de **potência**, dadas pelos arquivos "*sigproc/part2-signal-enroll_2-f08-54-16000Hz-preemph0.97-hamming#-magspec.png*" e "*sigproc/part2-signal-enroll_2-f08-54-16000Hz-preemph0.97-hamming#-powspec.png*". Existem imagens para 10 frames ao longo de todo o sinal.

4. Mostrar o banco de filtros em escala MEL, representado pela imagem "*features/part1-fbank-512-16000Hz.png*". Embora seja em escala MEL, a imagem mostra os filtros na escala Hertz. Na escala MEL todos os filtros tem o mesmo comprimento (que na escala Hertz crescem com a frequência). Explicar sobre o funcionamento da Cóclea no ouvido humano e como este banco de filtros simula sua funcionalidade (talvez colocar isso na seção anterior).
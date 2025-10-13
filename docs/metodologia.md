# Metodologia

1. **Pré-processamento/denoising** (Diffusion/GAN) para remover ruído e *ring artifacts* em slices 2D.
2. **Reconstrução 3D** condicional (Diffusion 3D/ControlNet) guiada por slices heterogêneos.
3. **Segmentação de poros e gargantas** (U-Net/3D-UNet + pós-processamento morfológico).
4. **Cálculo de propriedades** petrofísicas (porosidade, conectividade, parâmetros de permeabilidade proxy).
5. **Validação**: métricas de imagem e validação física (Φ, k, Vp/Vs quando aplicável).

> Reprodutibilidade: manter seeds, versões e *data cards*. Para dados grandes, usar Git LFS ou storage externo.

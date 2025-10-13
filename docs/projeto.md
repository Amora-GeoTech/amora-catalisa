# Visão Geral do Projeto

**Objetivo:** desenvolver uma ferramenta baseada em IA para **simulação e análise de rocha digital** (micro-CT), usando **modelos generativos avançados** (e.g., GAN, Diffusion) para **reconstrução 3D**, **segmentação**, e **estimativa de porosidade/estruturas internas**, como alternativa ao Avizo.

## Entregáveis (alto nível)
- Protótipo funcional (pipeline de reconstrução/segmentação/medidas).
- Métricas técnicas (IoU/DSC, SSIM, PVBT) e métricas físicas (Φ, k, conectividade).
- Relatórios mensais e documentação aberta.

## Diferenciais
- Condicionamento físico (rock physics) e validação petrofísica.
- Reconstrução 3D guiada por *slices* heterogêneos (ControlNet/condicionais).
- Integração com dados laboratoriais do LPS/UFPA.

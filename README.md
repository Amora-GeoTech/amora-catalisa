# AMORA Geotech Solutions — Catalisa ICT (Ciclo 2 · Etapa 2)

**Repositório oficial** da segunda fase do projeto no Catalisa ICT (Ciclo 2 · Etapa 2), focado em **I.A. para rocha digital e petrofísica**.  
Aqui centralizamos **código**, **documentação**, **relatórios** e **portfólio da equipe**.

---

## 🎯 Visão geral (o que queremos fazer)

Desenvolver um **software de análise e simulação de rocha digital** com **módulos integrados de petrofísica e dados de poço**, oferecendo uma **alternativa inovadora ao Avizo**.  
O produto combina **modelos generativos** e **pipelines reprodutíveis** para:

- **Reconstrução 3D** a partir de *slices* 2D (micro-CT), inclusive em cenários com amostragem esparsa;
- **Pré-processamento avançado** (remoção de ruído e *ring artifacts*, normalização e padronização de voxel);
- **Segmentação** (poros/matriz/gargantas) com métricas de imagem e métricas físicas;
- **Cálculo de propriedades petrofísicas** (ex.: porosidade volumétrica e métricas morfológicas relacionadas à permeabilidade);
- **Módulo de well logs** para ingestão, padronização, análises e geração de atributos/sinais sintéticos.

---

## 💡 Proposta de inovação (qual é a proposta)

1. **IA generativa aplicada à rocha digital**: uso de modelos **diffusion** (DDPM/CDDM) e **GANs** para:
   - reconstrução volumétrica 3D guiada por *slices*;
   - *denoising* e correção de artefatos;
   - *in-painting* e *super-resolution* em volumes micro-CT.
2. **Segmentação e validação física**: *baseline* com **U-Net (2D/3D)** e aprimoramentos com perdas estruturais; validação por **IoU/Dice/SSIM** e **métricas petrofísicas** (p.ex., porosidade da máscara).
3. **Integração petrofísica & well logs**: ingestão **LAS/CSV**, QA/QC, normalização de unidades, atributos derivados e **modelos supervisionados/não supervisionados** para classificação/estimação.
4. **Produtividade e reprodutibilidade**: interface amigável (Qt) + *notebooks* e **docs versionadas** (MkDocs), garantindo rastreabilidade e comparação com fluxos usuais do Avizo.

---

## 🧪 O que será aplicado (técnicas e stack)

- **Modelos de IA**: DDPM/CDDM (2D/3D), GANs (para *denoising*/super-res), **U-Net** (seg.) e **métricas** (IoU, Dice, SSIM).
- **Pipelines de micro-CT**: correção de *ring artifacts*, normalização de intensidade, **padronização de voxel size**, *tilt* e *bias*.
- **Petrofísica / Well logs**: ingestão **LAS**, controle de qualidade, atributos derivados (ex.: *feature engineering* para GR/RES/NPHI/DPHI), classificação/estimação e geração sintética de curvas.
- **Stack**: **Python** (NumPy, SciPy, scikit-image, PyTorch), **C++** (componentes de desempenho), **Qt Creator/Qt6** (UI), **MkDocs + Material** (docs), **GitHub Actions** (CI/CD).

> **Equipe & papéis nesta fase**  
> - **João Rafael B. S. da Silveira** — Pesquisador Proponente (coordenação técnica e científica)  
> - **José Frank** — **UI/UX** com **Qt Creator** (interface do módulo de rocha digital)  
> - **Marcus** e **Celso Rafael** — **Módulo de Well Logs** (ingestão, ETL, telas e análises)

---

## 🧩 MVP (Etapa 2 – o que entregaremos)

**Módulo 1 — Rocha Digital**
- Ingestão e organização de **volumes micro-CT** e metadados;
- **Pré-processamento** (denoise, *ring removal*, normalização);
- **Segmentação** (baseline U-Net) e **porosidade volumétrica**;
- *Notebooks* de validação com **IoU/Dice/SSIM** e comparação de cenários.

**Módulo 2 — Petrofísica & Well Logs**
- Ingestão **LAS/CSV**, QA/QC e padronização de unidades;
- Atributos deriváveis e visualizações rápidas;
- Protótipo de **modelos de classificação/estimação** e **logs sintéticos**.

---

## 🛣️ Roadmap (Catalisa ICT — Ciclo 2 · Etapa 2)

- **Mês 1**: kickoff e governança; arquitetura aprovada; ingestão/normalização; *denoising* inicial; UI (Qt) em evolução; *baseline* U-Net.  
- **Mês 2**: **Protótipo 2** (rocha digital + well logs) com fluxo ponta-a-ponta; métricas e *notebooks* de validação; ajustes de UI/UX.  
- **Mês 3**: otimizações (C++/GPU quando aplicável), documentação ampliada, comparativos com fluxos usuais do Avizo, empacotamento e *release*.

> *Acompanhamentos desta fase*: reuniões com consultor da Wylinka (ex.: Fellipe Fonseca) + checkpoints internos semanais.

---

## 📦 O que vem pronto (scaffold do repositório)

- **Site de documentação (MkDocs + Material)** com páginas: Projeto, Roadmap, Relatórios e Equipe.  
- **Template de Relatório Mensal** (Markdown) e **Issue Form** para coleta estruturada.  
- **GitHub Actions** para build e deploy automático do site no **GitHub Pages**.  
- **Licença MIT**, **Contributing**, **Code of Conduct** e **Citation**.

---

## 🚀 Como usar

1) Baixe este pacote inicial e extraia.  
2) (Opcional) Crie um ambiente Python e instale dependências:
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

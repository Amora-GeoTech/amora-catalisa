# AMORA Geotech Solutions — Catalisa ICT (Ciclo 2)

Repositório oficial do projeto para **gestão de relatórios**, **portfólio da equipe** e **documentação do produto** (IA para rocha digital).

## 📦 O que vem pronto
- **Site de documentação (MkDocs + Material)** com páginas: Projeto, Roadmap, Relatórios e Equipe.
- **Template de Relatório Mensal** (Markdown) e **Issue Form** para coleta estruturada.
- **GitHub Actions** para build e deploy automático do site no **GitHub Pages**.
- **Licença MIT**, **Contributing**, **Code of Conduct** e **Citation**.

## 🚀 Como usar
1. Baixe este pacote inicial e extraia.
2. (Opcional) Crie um ambiente Python e instale dependências:  
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Inicie o repositório e faça o primeiro push:
   ```bash
   git init
   git branch -M main
   git add .
   git commit -m "feat(repo): initial Catalisa repo scaffold"
   git remote add origin git@github.com:Amora-GeoTech/amora-catalisa.git
   git push -u origin main
   ```
4. Ative o **GitHub Pages** (Settings → Pages → Build and deployment: Source = GitHub Actions).

## 🗺 Estrutura
```
.
├─ .github/
│  ├─ ISSUE_TEMPLATE/relatorio-mensal.yml
│  └─ workflows/docs.yml
├─ docs/
│  ├─ index.md
│  ├─ projeto.md
│  ├─ metodologia.md
│  ├─ roadmap.md
│  ├─ equipe.md
│  └─ relatorios/
│     ├─ index.md
│     └─ 2025-10.md
├─ reports/RELATORIO_MENSAL_TEMPLATE.md
├─ mkdocs.yml
├─ requirements.txt
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
├─ CITATION.cff
├─ LICENSE
└─ .gitignore
```

## ✍️ Fluxo de relatórios
- Preencha `docs/relatorios/AAAA-MM.md` a cada mês **ou** abra um *Issue* usando o formulário “Relatório Mensal”.
- Faça *commit* e *push*. O site será atualizado automaticamente.

## 👥 Equipe
Edite `docs/equipe.md` para manter os perfis (nome, função, Lattes/LinkedIn, e-mail).

---

> Dica: para dados grandes (imagens micro-CT etc.), habilite **Git LFS**.

# Fish Pipeline

Este projeto implementa duas pipelines gémeas para contagem e medição de peixes a partir de imagens RGB-D.
A única diferença entre elas está na etapa final de medição: ponta-a-ponta vs skeleton.
O projeto foi construído com foco em organização, baixo acoplamento e testes unitários.

## Estrutura

```
src/
  fish_pipeline/
    data/            # Entidades, DTOs e validadores
    models/          # Modelos de densidade e segmentação (implementações dummy)
    steps/           # Etapas das pipelines (pré-processamento, QC, etc.)
    utils/           # Funções auxiliares de geometria e calibração
    pipelines/       # Pipeline de treino e pipeline de inferência
```

## Dependências

- Python 3.10+
- PyTest (opcional para testes)

Instalação (modo dev):

```bash
pip install -e .[dev]
```

## Testes

```bash
pytest
```

## Uso

Consulte `fish_pipeline/pipelines/inference.py` para um exemplo de execução das pipelines de inferência
(`EndToEndMeasurementPipeline` e `SkeletonMeasurementPipeline`).
O módulo `fish_pipeline/pipelines/training.py` demonstra a pipeline de treino do modelo de densidade.

# Tarefa: Implementar Concept Clustering Job

## Descrição
Desenvolver e implementar um job agendado para agrupar `(:Concept)` nodes relacionados em clusters temáticos. Este processo utilizará o **índice HNSW de embeddings de conceitos** (`docs/arch/on-concepts.md`, `docs/arch/on-vector_stores.md` e `docs/project/epic_1/sprint_5-hnswlib.md`) para identificar similaridades semânticas e formar grupos coerentes. Estes clusters serão a base para a posterior geração de páginas de assunto (`docs/arch/on-writing_vault_documents.md`, Seção "Subject Summary Agent"), contribuindo para as user stories de "Subject Pages" em `docs/project/product_definition.md`.

## Escopo

### Incluído
- Implementação de um job (executado pelo `aclarai-scheduler` de `docs/project/epic_1/sprint_3-Bootstrap_scheduler_and_vault_sync_job.md`) para agrupar conceitos relacionados.
- Desenvolvimento de um algoritmo de agrupamento que utilize os vetores do **índice HNSW de embeddings de conceitos** (`docs/arch/on-concepts.md`, seção "Concept Vector Index" e `docs/project/epic_1/sprint_5-hnswlib.md`) para formar clusters temáticos baseados em similaridade semântica. Os vetores de entrada serão provenientes do `concepts` vector store (`docs/arch/on-vector_stores.md`).
- Implementação de filtros para os clusters formados, incluindo:
    - **Tamanho mínimo de cluster** (`min_concepts`): filtrando grupos que contenham menos que um número configurável de conceitos, conforme `docs/arch/design_config_panel.md` (Seção 4, `subject_summaries`).
    - **Limiar de similaridade** (`similarity_threshold`): utilizando um limiar de distância de cosseno (ou métrica equivalente) para determinar a coesão interna dos clusters, conforme `docs/arch/design_config_panel.md` (Seção 4, `subject_summaries`).
    - **Tamanho máximo de cluster** (`max_concepts`): limitando o número de conceitos por grupo para evitar clusters excessivamente grandes, conforme `docs/arch/design_config_panel.md` (Seção 4, `subject_summaries`).
- Armazenamento em cache das atribuições de grupo (qual conceito pertence a qual cluster) para uso posterior pelo **Agente de Resumo de Assunto** (`docs/project/epic_1/sprint_10-Implement_Subject_Summary_Agent.md`). Este cache pode ser em memória ou em um armazenamento persistente leve.
- Integração do job com o `aclarai-scheduler`, permitindo sua configuração e agendamento via `aclarai.config.yaml` e a UI, conforme `docs/project/epic_1/sprint_6-Add_configuration_controls.md`.
- Documentação clara do processo de clustering, da escolha do algoritmo e da configuração.
- Implementação de testes para verificar a correta formação de clusters e a aplicação dos filtros.

### Excluído
- Interface de usuário para visualização interativa de clusters (será implementada em tarefa separada, possivelmente em Sprint 10).
- Geração de páginas de assunto (`[[Subject:XYZ]]` Markdown files) (será implementada na tarefa "Implementar Agente de Resumo de Assunto" de Sprint 10, que **consumirá o output deste job**).
- Otimizações avançadas de desempenho que vão além de um algoritmo de clusterização eficiente e a leitura otimizada do índice HNSW.
- Análise avançada de relações entre clusters (o foco é na formação de clusters básicos para geração de sumários).

## Critérios de Aceitação
- O job utiliza corretamente os vetores do **índice HNSW de embeddings de conceitos** (`docs/arch/on-concepts.md`, `docs/arch/on-vector_stores.md`) para formar clusters.
- Os clusters formados são semanticamente coerentes e temáticos, refletindo as similaridades entre os `(:Concept)` nodes (`docs/arch/on-concepts.md`).
- Os filtros por tamanho mínimo de cluster (`min_concepts`), tamanho máximo de cluster (`max_concepts`) e limiar de similaridade (`similarity_threshold`) funcionam adequadamente, conforme configurado em `aclarai.config.yaml` (`docs/arch/design_config_panel.md`).
- As atribuições de grupo (mapeamento `concept_id` para `cluster_id`) são armazenadas em cache de forma acessível para uso posterior pelo Agente de Resumo de Assunto.
- A integração com o `aclarai-scheduler` (`docs/project/epic_1/sprint_3-Bootstrap_scheduler_and_vault_sync_job.md`) funciona corretamente, permitindo que o job seja agendado e executado.
- A documentação clara do processo de clustering e configuração está disponível no repositório.
- Testes automatizados demonstram a funcionalidade e robustez do job, cobrindo a correta formação de clusters e a aplicação dos filtros.

## Dependências
- **Índice HNSW de embeddings de conceitos** (`docs/arch/on-concepts.md`, `docs/arch/on-vector_stores.md`) implementado e populado com os vetores dos `(:Concept)` nodes (`docs/project/epic_1/sprint_5-hnswlib.md`).
- `(:Concept)` nodes existentes no Neo4j, com embeddings atualizados (`docs/project/epic_1/sprint_5-Create_Update_Tier_3.md` e `docs/project/epic_1/sprint_5-Refresh_embeddings.md`).
- Sistema de agendamento (`aclarai-scheduler` com `APScheduler`) implementado e funcional (`docs/project/epic_1/sprint_3-Bootstrap_scheduler_and_vault_sync_job.md`).
- Configuração clara dos parâmetros de clustering em `settings/aclarai.config.yaml` (`docs/arch/design_config_panel.md`).

## Entregáveis
- Código-fonte do job de clustering de conceitos (dentro do serviço `aclarai-scheduler`).
- Implementação do algoritmo de formação de clusters.
- Sistema de cache para atribuições de grupo.
- Documentação técnica do processo e configuração.
- Testes unitários e de integração.

## Estimativa de Esforço
- 3 dias de trabalho

## Riscos e Mitigações
- **Risco**: Clusters formados não são semanticamente coerentes ou significativos para o usuário.
  - **Mitigação**: Calibrar cuidadosamente os limiares de similaridade e os parâmetros de tamanho de cluster (`min_concepts`, `max_concepts`) com base em `docs/arch/design_config_panel.md`. Implementar validação manual de amostras de clusters para verificar a relevância semântica e ajustar os parâmetros do algoritmo.
- **Risco**: Desempenho inadequado com um grande número de `(:Concept)` nodes, impactando o tempo de execução do job.
  - **Mitigação**: Implementar otimizações de algoritmo para a fase de clusterização. Considerar abordagens incrementais ou a execução em intervalos de tempo menos frequentes para grandes volumes de dados. Otimizar a recuperação de embeddings do vector store.
- **Risco**: Formação de clusters muito pequenos (muito granular) ou muito grandes (pouco específico).
  - **Mitigação**: Utilizar os parâmetros configuráveis para tamanho mínimo e máximo de cluster (`min_concepts`, `max_concepts`) em `docs/arch/design_config_panel.md` para controlar a granularidade. A escolha do algoritmo de clustering também influenciará (e.g., DBSCAN pode gerar mais outliers, agrupamento hierárquico pode ser controlado por corte na árvore).

## Notas Técnicas
- O job deve ser executado periodicamente pelo `aclarai-scheduler`, lendo os parâmetros de `settings/aclarai.config.yaml` para `subject_summaries` (`similarity_threshold`, `min_concepts`, `max_concepts`).
- Considerar o uso de algoritmos de clusterização como DBSCAN (útil para lidar com densidade e identificar outliers) ou agrupamento hierárquico (útil para explorar diferentes granularidades de cluster) para a formação de clusters. A escolha pode depender da complexidade e distribuição dos embeddings.
- Implementar métricas de qualidade (e.g., Coeficiente de Silhueta, Índice de Davies-Bouldin) para avaliar a coerência e separação dos clusters formados. Isso pode auxiliar na calibração dos parâmetros.
- O sistema de cache para atribuições de grupo deve ser projetado para permitir rápida recuperação e atualização incremental de associações de conceitos a clusters. Pode ser um simples dicionário em memória para o MVP, mas com consideração para persistência em caso de reinício.
- O job deve lidar com conceitos outliers que não se encaixam bem em nenhum cluster, garantindo que eles não sejam perdidos no processo, embora não formem um "assunto" por si só.
- Este job é um pré-requisito direto para a tarefa "Implementar Subject Summary Agent" de Sprint 10, que consumirá esses clusters.

**Nota Técnica:** A implementação deste job deve recuperar os embeddings de todos os conceitos canônicos de uma só vez, utilizando um método de recuperação em lote (`get_embeddings_for_concepts`) do vector store. O uso de `similarity_search` em um loop para cada conceito é um anti-padrão que deve ser evitado para garantir o desempenho.

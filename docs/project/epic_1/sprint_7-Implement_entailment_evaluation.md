# Tarefa: Implementar Agente de Avaliação de Entailment

## Descrição
Desenvolver e implementar um agente de avaliação de entailment que analise a relação lógica entre claims (`(:Claim)` nodes) e suas fontes (`:Block` nodes), produzindo uma pontuação (`entailed_score`) que indique o grau de suporte lógico da fonte para o claim. Esta pontuação é crucial para o processo de avaliação de qualidade dos claims no aclarai, conforme detalhado em `docs/arch/on-evaluation_agents.md` e `docs/project/technical_overview.md`.

## Escopo

### Incluído
- Implementação do **Agente de Avaliação de Entailment**, seguindo a descrição em `docs/arch/on-evaluation_agents.md` (Seção "Agent: `entailment`").
- Desenvolvimento de lógica para analisar a `source` (bloco Markdown original ou contexto estruturado) e o `claim` (texto do claim candidato), produzindo um `entailed_score` (float entre 0 e 1). A estrutura de entrada para o agente é definida em `docs/arch/on-evaluation_agents.md` (Seção "Input Format").
- Armazenamento da pontuação `entailed_score` como uma propriedade na aresta `[:ORIGINATES_FROM]` que conecta o `(:Claim)` node ao seu `(:Block)` de origem no Neo4j, conforme `docs/arch/on-evaluation_agents.md` (Seção "Storage").
- Armazenamento da pontuação `entailed_score` como metadado em um comentário HTML no Markdown Tier 1 (`<!-- aclarai:entailed_score=0.91 -->`), conforme `docs/arch/on-evaluation_agents.md` (Seção "Storage").
- **Utilização da lógica de escrita atômica para arquivos Markdown** (implementada em Sprint 3, detalhada em `docs/arch/on-filehandle_conflicts.md`) para a atualização dos metadados no Markdown Tier 1.
- Implementação de um sistema de retry robusto para o agente em casos de falha (e.g., timeout, erro do LLM).
- Tratamento adequado de valores `null` para `entailed_score` em caso de falha do agente após os retries. Claims com scores `null` não serão escritos em Markdown e não serão vinculados a conceitos, conforme `docs/arch/on-evaluation_agents.md` (Seção "Failure Handling").
- Documentação clara do processo de avaliação, incluindo a estrutura do prompt, a saída esperada e a interpretação das pontuações.
- Implementação de testes para verificar a correta avaliação do `entailed_score` em diversos cenários (suporte forte, fraco, contradição, não-relacionado).

### Excluído
- Interface de usuário para visualização *direta* das pontuações (isso será coberto em Sprint 8).
- Otimizações avançadas de desempenho que vão além de um sistema de retry eficiente e o uso de prompts otimizados.
- Treinamento de modelos personalizados para avaliação de entailment (foco no uso de LLMs configuráveis, conforme `docs/arch/on-evaluation_agents.md`, Seção "Model Configuration").
- Processamento em lote de volumes *extremamente* grandes de dados (foco na funcionalidade e correção).
- Integração com sistemas externos de verificação de fatos ou APIs de NLI de terceiros (o LLM é a fonte principal de entailment).

## Critérios de Aceitação
- O Agente de Entailment está implementado e analisa corretamente a relação lógica entre source e claim.
- A pontuação `entailed_score` é produzida com precisão e consistência para diferentes inputs.
- A pontuação `entailed_score` é armazenada corretamente na aresta `[:ORIGINATES_FROM]` no grafo Neo4j e como metadado no Markdown Tier 1.
- **A atualização do Markdown Tier 1 com os metadados de pontuação utiliza a lógica de escrita atômica de forma robusta e segura.**
- **Os marcadores `aclarai:id` e `ver=` existentes nos blocos Markdown Tier 1 são preservados e a propriedade `ver=` é incrementada quando os metadados de pontuação são adicionados/atualizados.**
- O sistema de retry funciona adequadamente para casos de falha do agente, com o `entailed_score` sendo definido como `null` após falhas persistentes.
- O tratamento de valores `null` está apropriado, garantindo que claims com `null` score não sejam processados downstream (no que diz respeito a serem escritos em Markdown ou vinculados a conceitos *nesta fase*).
- A documentação clara do processo de avaliação de entailment, incluindo a estrutura do prompt e a interpretação da pontuação, está disponível.
- Testes automatizados demonstram a funcionalidade e robustez do agente, incluindo casos de sucesso e falha.

## Dependências
- Pipeline Claimify implementado (de Sprint 3), fornecendo claims e suas fontes.
- Nós `(:Claim)` e `(:Block)` e arestas `[:ORIGINATES_FROM]` criados no Neo4j (de Sprint 3), para armazenamento das pontuações.
- Acesso ao sistema de arquivos para atualização de metadados Markdown Tier 1.
- **Fábrica de Ferramentas Compartilhada (`ToolFactory`) implementada** (de `sprint_7-Implement_Shared_Tool_Factory.md`), que fornecerá as ferramentas necessárias (e.g., `Neo4jQueryTool`, `VectorSearchTool`).
- Modelo de linguagem (LLM) configurado para o **agente interno**, conforme `docs/arch/design_config_panel.md`.
- Mecanismos de escrita atômica para arquivos Markdown (de Sprint 3).

## Entregáveis
- Código-fonte do Agente de Avaliação de Entailment.
- Lógica para armazenamento de pontuações na aresta `[:ORIGINATES_FROM]` no Neo4j e em metadados Markdown.
- Implementação do sistema de retry e tratamento de `null` values para o score.
- Documentação detalhada do processo de avaliação de entailment.
- Testes unitários e de integração para o agente de avaliação.

## Estimativa de Esforço
- 3 dias de trabalho

## Riscos e Mitigações
- **Risco**: Inconsistência nas avaliações do LLM devido a variações no modelo ou prompt.
  - **Mitigação**: Calibrar prompts cuidadosamente, utilizando exemplos few-shot se necessário. Implementar testes de regressão com um conjunto fixo de "golden answers" para monitorar a consistência do `entailed_score`.
- **Risco**: Falhas frequentes no processamento do LLM, resultando em muitos scores `null`.
  - **Mitigação**: Implementar um sistema robusto de retry com backoff exponencial. Monitorar as taxas de falha do LLM e ajustar o modelo ou o tamanho/complexidade do prompt.
- **Risco**: Pontuações numéricas produzidas pelo LLM não refletem adequadamente a qualidade real de entailment.
  - **Mitigação**: Validar a correlação das pontuações do LLM com uma amostra de avaliações manuais (humanas) e ajustar o prompt ou o mapeamento de saída do LLM para a escala de 0-1 conforme necessário.

## Notas Técnicas
- O `entailed_score` deve ser um `Float` que pode ser `null` no Neo4j e no Markdown, conforme `docs/arch/on-evaluation_agents.md`.
- A estrutura do prompt para o LLM deve ser otimizada para a tarefa de entailment, conforme o exemplo em `docs/arch/on-evaluation_agents.md` (Seção "Prompt structure").
- O agente será implementado como um `CodeActAgent` do LlamaIndex (ou similar), que utiliza um conjunto de ferramentas fornecido pela `ToolFactory` para coletar contexto antes de tomar uma decisão.
- O logging deve incluir não apenas o `claim_id` e o `source_id`, mas também as ferramentas utilizadas e os resultados observados durante o ciclo de raciocínio do agente.
- A escrita dos metadados no Markdown deve ser feita utilizando a lógica de escrita atômica já existente para garantir a segurança dos arquivos.
- O logging deve incluir o `claim_id` e o `source_id` para facilitar a depuração e rastreabilidade.

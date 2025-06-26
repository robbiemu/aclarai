# Tarefa: Implementar Agente de Avaliação de Coverage

## Descrição
Desenvolver e implementar um agente de avaliação de coverage que analise a completude dos claims em relação às suas fontes, produzindo uma pontuação (`coverage_score`) que indique o quanto o claim captura os elementos verificáveis da fonte. Este agente também identificará e registrará elementos omitidos, enriquecendo o grafo de conhecimento para futuras análises, conforme detalhado em `docs/arch/on-evaluation_agents.md` e `docs/project/technical_overview.md`.

## Escopo

### Incluído
- Implementação do **Agente de Avaliação de Coverage**, seguindo a descrição em `docs/arch/on-evaluation_agents.md` (Seção "Agent: `coverage`").
- Desenvolvimento de lógica para calcular o `coverage_score` (float entre 0.0 e 1.0) a partir do `claim` (texto do claim candidato) e sua `source` (bloco Markdown original ou contexto estruturado). A estrutura de entrada para o agente é definida em `docs/arch/on-evaluation_agents.md` (Seção "Input Format").
- **Utilização de ferramentas da `ToolFactory`**, como a `WebSearchTool` (se configurada), para verificar a significância de entidades e melhorar a precisão da avaliação de cobertura.
- **Extração de elementos verificáveis omitidos** que estavam na `source` mas não foram capturados pelo `claim`, conforme `docs/arch/on-evaluation_agents.md`.
- **Adição de nós `(:Element)` ao grafo Neo4j** para cada elemento verificável omitido, com a propriedade `text` (conforme `docs/arch/on-evaluation_agents.md`).
- **Criação de arestas `[:OMITS]` entre o `(:Claim)` node e cada `(:Element)` node** omitido no grafo, conforme `docs/arch/on-evaluation_agents.md` (Seção "Missed Elements").
- Armazenamento da pontuação `coverage_score` como uma propriedade na aresta `[:ORIGINATES_FROM]` que conecta o `(:Claim)` node ao seu `(:Block)` de origem no Neo4j, conforme `docs/arch/on-evaluation_agents.md` (Seção "Storage").
- Armazenamento da pontuação `coverage_score` como metadado em um comentário HTML no Markdown Tier 1 (`<!-- aclarai:coverage_score=0.77 -->`), conforme `docs/arch/on-evaluation_agents.md` (Seção "Storage").
- **Utilização da lógica de escrita atômica para arquivos Markdown** (implementada em Sprint 3, detalhada em `docs/arch/on-filehandle_conflicts.md`) para a atualização dos metadados no Markdown Tier 1.
- Implementação de um sistema de retry robusto para o agente em casos de falha (e.g., timeout, erro do LLM), consistente com o tratamento de falhas de outros agentes de avaliação (conforme `docs/arch/on-evaluation_agents.md`, Seção "Failure Handling").
- Tratamento adequado de valores `null` para `coverage_score` em caso de falha do agente após os retries. Claims com scores `null` não serão escritos em Markdown nem vinculados a conceitos (conforme `docs/arch/on-evaluation_agents.md`, Seção "Failure Handling").
- Documentação clara do processo de avaliação, incluindo a estrutura do prompt, a saída esperada e a interpretação das pontuações e dos elementos omitidos.
- Implementação de testes para verificar a correta avaliação do `coverage_score` e a extração/persistência de elementos omitidos em diversos cenários.

### Excluído
- Interface de usuário para visualização *direta* das pontuações ou dos elementos omitidos (isso será coberto em Sprint 8).
- Otimizações avançadas de desempenho que vão além de um sistema de retry eficiente e o uso de prompts otimizados.
- Treinamento de modelos personalizados para avaliação de coverage (foco no uso de LLMs configuráveis, conforme `docs/arch/on-evaluation_agents.md`, Seção "Model Configuration").
- Processamento em lote de volumes *extremamente* grandes de dados (foco na funcionalidade e correção).
- Integração com sistemas externos de análise de texto ou verificação de fatos (o LLM é a fonte principal de avaliação).

## Critérios de Aceitação
- O Agente de Coverage está implementado e calcula corretamente o `coverage_score` a partir do claim e da source.
- Elementos verificáveis omitidos são extraídos com precisão e consistência.
- Nós `(:Element)` e arestas `[:OMITS]` são adicionados corretamente ao grafo Neo4j, com as propriedades definidas.
- A pontuação `coverage_score` é armazenada corretamente na aresta `[:ORIGINATES_FROM]` no grafo Neo4j e como metadado no Markdown Tier 1.
- **A atualização do Markdown Tier 1 com os metadados de pontuação utiliza a lógica de escrita atômica de forma robusta e segura.**
- **Os marcadores `aclarai:id` e `ver=` existentes nos blocos Markdown Tier 1 são preservados e a propriedade `ver=` é incrementada quando os metadados de pontuação são adicionados/atualizados.**
- O sistema de retry funciona adequadamente para casos de falha do agente, com o `coverage_score` sendo definido como `null` após falhas persistentes.
- O tratamento de valores `null` está apropriado, garantindo que claims com `null` score não sejam processados downstream (no que diz respeito a serem escritos em Markdown ou vinculados a conceitos *nesta fase*).
- A documentação clara do processo de avaliação de coverage, incluindo a estrutura do prompt e a interpretação da pontuação e dos elementos omitidos, está disponível.
- Testes automatizados demonstram a funcionalidade e robustez do agente, incluindo casos de sucesso e falha, e a correta extração/persitência de elementos.

## Dependências
- Pipeline Claimify implementado (de Sprint 3), fornecendo claims e suas fontes.
- Nós `(:Claim)` e `(:Block)` e arestas `[:ORIGINATES_FROM]` criados no Neo4j (de Sprint 3), para armazenamento das pontuações.
- Acesso ao Neo4j para criação de novos nós `(:Element)` e arestas `[:OMITS]`.
- Acesso ao sistema de arquivos para atualização de metadados Markdown Tier 1.
- **Fábrica de Ferramentas Compartilhada (`ToolFactory`) implementada** (de `sprint_7-Implement_Shared_Tool_Factory.md`), que fornecerá as ferramentas necessárias, incluindo a `WebSearchTool` condicional.
- Modelo de linguagem (LLM) configurado para o **agente interno**, conforme `docs/arch/design_config_panel.md`.
- Mecanismos de escrita atômica para arquivos Markdown (de Sprint 3).

## Entregáveis
- Código-fonte do Agente de Avaliação de Coverage.
- Lógica para calcular `coverage_score` e extrair elementos omitidos.
- Lógica para armazenamento de pontuações e criação de nós `(:Element)` e arestas `[:OMITS]` no Neo4j.
- Implementação do sistema de retry e tratamento de `null` values para o score.
- Documentação detalhada do processo de avaliação de coverage.
- Testes unitários e de integração para o agente de avaliação.

## Estimativa de Esforço
- 3 dias de trabalho

## Riscos e Mitigações
- **Risco**: Dificuldade em identificar todos os elementos verificáveis relevantes na source ou em determinar a completude do claim pelo LLM.
  - **Mitigação**: Otimizar a estrutura do prompt do LLM para guiar a análise de completude, utilizando exemplos few-shot se necessário. Realizar testes de regressão com um conjunto de dados anotado para validar a precisão do `coverage_score` e a extração dos elementos omitidos.
- **Risco**: Falhas frequentes no processamento do LLM, resultando em muitos scores `null` ou extrações incompletas de elementos.
  - **Mitigação**: Implementar um sistema robusto de retry com backoff exponencial. Monitorar as taxas de falha do LLM e ajustar o modelo ou o tamanho/complexidade do prompt.
- **Risco**: Pontuações numéricas produzidas pelo LLM não refletem adequadamente a completude real.
  - **Mitigação**: Validar a correlação das pontuações do LLM com uma amostra de avaliações manuais (humanas) e ajustar o prompt ou o mapeamento de saída do LLM para a escala de 0-1 conforme necessário.

## Notas Técnicas
- O `coverage_score` deve ser um `Float` que pode ser `null` no Neo4j e no Markdown, conforme `docs/arch/on-evaluation_agents.md`.
- A estrutura do prompt para o LLM deve ser otimizada para a tarefa de coverage, instruindo o LLM a identificar e listar os elementos omitidos.
- O agente será implementado como um `CodeActAgent` do LlamaIndex (ou similar). Ele **deve funcionar graciosamente** se a `WebSearchTool` não estiver disponível, baseando sua análise no contexto interno do grafo e dos vetores.
- A estrutura do prompt para o LLM deve instruí-lo a usar as ferramentas disponíveis para verificar a importância de entidades antes de decidir se sua omissão impacta a pontuação de cobertura.
- A escrita dos metadados no Markdown deve ser feita utilizando a lógica de escrita atômica já existente para garantir a segurança dos arquivos.
- O logging deve incluir o `claim_id`, `source_id`, `coverage_score` e, se aplicável, os `Element`s omitidos para facilitar a depuração e rastreabilidade.
- A criação de nós `(:Element)` e arestas `[:OMITS]` deve ser feita em lote para otimizar a performance de escrita no Neo4j.

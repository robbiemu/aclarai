# Tarefa: Implementar Fábrica de Ferramentas Compartilhada para Agentes

## Descrição
Desenvolver uma `ToolFactory` centralizada que forneça aos agentes um conjunto padrão de ferramentas (e.g., consulta Neo4j, busca vetorial, busca na web) com base na configuração do sistema em `settings/aclarai.config.yaml`. Esta fábrica irá abstrair a inicialização de ferramentas individuais, tornando o desenvolvimento de agentes mais limpo, consistente e configurável.

## Escopo

### Incluído
- Criação de uma classe `ToolFactory` dentro do diretório `shared/aclarai_shared/tools/`.
- A fábrica lerá uma nova seção `tools:` no `settings/aclarai.config.yaml` para determinar quais ferramentas habilitar e como configurá-las.
- Implementação de um método `get_tools_for_agent(agent_name: str)` que retorna uma lista de instâncias de `Tool` do LlamaIndex configuradas.
- **Implementação das seguintes ferramentas principais:**
    - `Neo4jQueryTool`: Executa consultas Cypher no banco de dados Neo4j.
    - `VectorSearchTool`: Realiza buscas de similaridade semântica nos vector stores do projeto (e.g., `utterances`, `concepts`).
    - **`WebSearchTool`**: Uma ferramenta plugável para buscas externas na web. Deve suportar múltiplos provedores via configuração.
        - Implementar suporte para pelo menos um provedor (e.g., Tavily) como referência.
- **Definição do esquema de configuração** no `aclarai.config.default.yaml` para a ferramenta de busca na web, incluindo:
    - `tools.web_search.provider`: e.g., "tavily", "brave", "serper".
    - `tools.web_search.api_key_env_var`: O *nome* da variável de ambiente que contém a chave da API (e.g., `TAVILY_API_KEY`). A fábrica lerá esta variável para inicializar a ferramenta.
- Se o provedor de busca na web não estiver configurado ou a variável de ambiente da chave da API não estiver definida, a fábrica **não** criará ou retornará a `WebSearchTool`.
- Garantir que os agentes (como os agentes de avaliação) possam ser inicializados com a lista de ferramentas fornecida por esta fábrica.
- Documentação da `ToolFactory`, do esquema de configuração para ferramentas e de como adicionar novos provedores de ferramentas.

### Excluído
- Implementação dos agentes em si (eles são consumidores desta fábrica).
- Uma UI para configurar ferramentas (a configuração é via YAML/env vars apenas).
- Implementação de todos os provedores de busca na web possíveis (um ou dois são suficientes para provar o padrão).

## Critérios de Aceitação
- A `ToolFactory` lê corretamente a seção `tools:` da configuração.
- A fábrica fornece instâncias de `Neo4jQueryTool` e `VectorSearchTool` quando chamada.
- A fábrica **apenas** fornece uma `WebSearchTool` se um provedor e sua variável de ambiente de chave de API correspondente estiverem configurados.
- A `WebSearchTool` é funcional e pode executar uma busca na web.
- Os agentes podem ser inicializados com sucesso com a lista de ferramentas retornada pela fábrica.
- A implementação é extensível, permitindo que novas ferramentas ou provedores de busca na web sejam adicionados com alterações mínimas no código.

## Dependências
- Sistema de configuração principal (`aclarai.config.yaml`).
- Biblioteca LlamaIndex para `BaseTool` e integrações de agentes.

## Entregáveis
- Código para a `ToolFactory` e suas ferramentas suportadas em `shared/aclarai_shared/tools/`.
- `aclarai.config.default.yaml` atualizado com a nova seção `tools:`.
- Testes unitários e de integração para a fábrica e cada ferramenta.
- Documentação do sistema de ferramentas.

## Estimativa de Esforço
- 2 dias de trabalho

## Riscos e Mitigações
- **Risco**: Configuração incorreta de ferramentas levando a falhas em tempo de execução nos agentes.
  - **Mitigação**: Implementar validação robusta da configuração e logging claro quando uma ferramenta não pode ser inicializada. Os agentes devem ser capazes de funcionar com um subconjunto de ferramentas, se possível.
- **Risco**: Dependências de API de terceiros (para busca na web) podem mudar ou se tornar instáveis.
  - **Mitigação**: Projetar a `WebSearchTool` de forma plugável, facilitando a troca ou adição de novos provedores. Implementar tratamento de erros e retries para chamadas de API.
- **Risco**: Vazamento de chaves de API através de logs ou erros.
  - **Mitigação**: Assegurar que as chaves de API nunca sejam registradas em log. A fábrica deve apenas ler os nomes das variáveis de ambiente, não seus valores, passando-os diretamente para os SDKs das ferramentas.

## Notas Técnicas
- A `ToolFactory` deve ser uma classe singleton ou instanciada uma vez por serviço para evitar a reinicialização repetida das ferramentas.
- As ferramentas devem ser projetadas para serem thread-safe se forem usadas em um ambiente concorrente.
- A documentação deve incluir instruções claras sobre como configurar cada provedor de busca na web suportado, incluindo qual variável de ambiente definir.
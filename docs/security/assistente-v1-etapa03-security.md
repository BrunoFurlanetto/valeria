# Security Review - Assistente V1 Etapa 03

Data: 2026-05-03
Agente: Ana Ferreira (security)
Branch: feature/assistente-v1
Status: revalidado apos correcao de SEC-03-001

## Escopo revisado

- `app/assistant/core.py`
- `app/assistant/event_logger.py`
- `app/assistant/gemini_client.py`
- `app/assistant/voice_loop.py`
- `app/assistant/cli.py`
- `tests/test_assistant_stage3.py`
- `tests/test_assistant_cli.py`
- `tests/test_voice_flow_v1.py`

## Resumo executivo

Revalidacao focada no bloqueador SEC-03-001: fallbacks de `AssistantCore` nao devem ecoar entrada/transcript bruto para stdout nem para TTS; logs nao devem conter transcript, resposta ou segredos; testes devem cobrir o comportamento seguro.

Resultado: sem bloqueadores restantes para a Etapa 03. SEC-03-001 foi corrigido e coberto por testes. SEC-03-002 permanece risco baixo e nao bloqueante.

## Findings

### SEC-03-001 - Resolvido - Fallback nao ecoa entrada do usuario

Arquivo: `app/assistant/core.py`

Estado anterior: quando Gemini estava indisponivel, falhava ou retornava vazio, `_fallback_response()` montava uma resposta contendo o `user_text` completo. Essa resposta podia chegar ao stdout pelo modo texto e ao TTS pelo modo voz.

Estado revalidado:

- `_fallback_response()` agora usa mensagens genericas para `gemini_unavailable`, `gemini_error` e `empty_response`.
- O parametro `user_text` ainda existe na assinatura, mas nao e usado para montar a resposta.
- `Assistant.respond()` continua retornando a string da resposta; nos fallbacks revisados, essa string nao contem a entrada bruta.
- No fluxo de voz, o TTS recebe `response_text`; para os fallbacks de `AssistantCore`, esse texto agora e generico e nao contem transcript bruto.

Testes ajustados:

- `tests/test_assistant_stage3.py` verifica que `ola` e `abrir agenda` nao aparecem nos fallbacks.
- `tests/test_assistant_cli.py` verifica que `Valeria recebeu: ola` nao aparece no stdout do modo texto.
- `tests/test_voice_flow_v1.py` verifica que transcript e resposta completos nao sao impressos no fluxo de voz e que o fallback antigo nao aparece no modo texto sem PyAudio.

### SEC-03-002 - Baixo - Log JSONL append-only sem rotacao ou limite de tamanho

Arquivo: `app/assistant/event_logger.py`

Permanece baixo e nao bloqueante. O logger grava JSONL em append-only e nao possui rotacao, limite de tamanho ou retencao. O conteudo gravado e allowlistado e nao inclui transcript, resposta, prompt ou segredos, entao o risco principal e churn/disco em execucao continua.

Controles existentes:

- caminho padrao em `tmp/valeria-logs/events.jsonl`;
- payload allowlistado em `_sanitize_event()`;
- apenas metadados operacionais sao persistidos: timestamp, evento, status, modelo, fallback, tipo de erro, latencia, tamanhos e tamanho do historico;
- falhas de escrita nao quebram o fluxo.

Recomendacao futura:

- adicionar limite simples de tamanho, rotacao ou limpeza periodica antes de uso continuo por longos periodos.

## Controles verificados

- Fallbacks: nao ha eco de `user_text` nas mensagens genericas de erro/indisponibilidade/resposta vazia.
- Stdout: o modo texto imprime `Assistant.respond()`, mas os fallbacks revisados nao contem entrada bruta; o fluxo de voz imprime apenas mensagens de estado.
- TTS: `VoiceLoop` envia `response_text` para `synthesize_speech`; em fallback local revisado, `response_text` nao contem transcript bruto.
- Logs: `EventLogger` persiste somente campos allowlistados e ignora `transcript`, `response`, `GOOGLE_API_KEY` e `api_key`.
- Erros Gemini: `_generate_content()` encapsula falhas como `GeminiClientError("{operation} request failed.")`, sem detalhe bruto do provider.
- Segredos: busca por padroes comuns encontrou apenas strings de teste/relatorio, sem segredo real versionado detectado.
- Historico: `AssistantCore.history` permanece em memoria e limitado por `max_history_messages`.

## Checks executados

- `git status --short --branch`
- `git diff -- app\assistant\core.py app\assistant\event_logger.py app\assistant\gemini_client.py tests\test_assistant_stage3.py tests\test_assistant_cli.py tests\test_voice_flow_v1.py`
- `venv\Scripts\python.exe -m pytest -q` -> 38 passed
- `venv\Scripts\python.exe -m pytest tests\test_assistant_stage3.py tests\test_assistant_cli.py tests\test_voice_flow_v1.py -q` -> 37 passed
- `rg -n 'Valeria recebeu|recebeu:|\{user_text\}|f".*user_text|format\(user_text|print\(.*transcript|print\(.*response_text|synthesize_speech\(.*transcript' app\assistant tests`
- `rg -n 'assert .*not in (response|response\.text|captured\.out|combined_output|combined)' tests\test_assistant_stage3.py tests\test_assistant_cli.py tests\test_voice_flow_v1.py`
- `rg -n "AIza[0-9A-Za-z_-]{20,}|github_pat_|ghp_|sk-[A-Za-z0-9]|token=|password|secret|BEGIN [A-Z ]*PRIVATE KEY" .`

Observacao sobre buscas:

- A busca por eco antigo ainda encontra o texto `Valeria recebeu` em asserts negativos e neste relatorio; nao encontrou uso produtivo do fallback antigo.
- A busca por segredos encontrou apenas fixtures/strings sinteticas em testes e este relatorio.

## Recomendacao final

Aprovar Security da Etapa 03. Nao ha bloqueadores restantes apos a correcao de SEC-03-001. Manter SEC-03-002 como backlog tecnico baixo e nao bloqueante antes de execucao continua.

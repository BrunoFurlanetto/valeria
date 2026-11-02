# Security Audit — T-001: Trocar provedores Gemini para Groq + ElevenLabs

**Agente:** ana-ferreira-security
**Data:** 2026-11-02
**Branch:** feature/T-001-trocar-provedores-groq-elevenlabs
**Status:** APROVADO — sem vulnerabilidades criticas

---

## Escopo auditado

Todos os arquivos criados ou alterados na feature T-001:

- `app/assistant/groq_chat_client.py`
- `app/assistant/groq_stt_client.py`
- `app/assistant/elevenlabs_tts_client.py`
- `app/assistant/feedback_player.py`
- `app/assistant/messages.py`
- `app/assistant/config.py`
- `app/assistant/core.py`
- `app/assistant/voice_loop.py`
- `app/assistant/event_logger.py`
- `scripts/generate_static_audio.py`
- `app/requirements.txt`
- `.env.example`
- `Dockerfile`
- `.gitignore`

---

## Checklist de seguranca

### Autenticacao e autorizacao
- [x] Todos os clientes API requerem chave valida na inicializacao (`GroqChatClient`, `GroqSttClient`, `ElevenLabsTtsClient` levantam erro se `api_key` vazio)
- [N/A] Endpoints internos nao acessiveis por roles nao autorizados — aplicacao CLI local, sem endpoints HTTP
- [N/A] Escalacao de privilegio via manipulacao de parametros — sem autenticacao multi-role

### Dados e exposicao
- [x] Nenhum dado sensivel exposto em logs — `EventLogger` usa allowlist (`status`, `model`, `fallback`, `error_type`, `latency_ms`, `input_length`, `response_length`, `history_size`, `tokens_used`); transcript, resposta e segredos NUNCA entram no log
- [x] Chaves API armazenadas como atributos de instancia (`_api_key`) sem exposicao em `__repr__`, logs ou print
- [x] Mensagens de erro nao revelam detalhes internos — erros dos clientes usam mensagens genericas ("Transcricao falhou.", "TTS falhou: {tipo_excecao}", "Todos os modelos de chat falharam.")

### Inputs e superficie de ataque
- [x] Respostas de APIs Groq e ElevenLabs tratadas como dados estruturados via SDK — sem `eval()`, `exec()` ou concatenacao insegura
- [x] Audio do ElevenLabs escrito como bytes brutos em arquivo WAV — sem parsing inseguro
- [x] Texto do Groq extraido via `response.choices[0].message.content` — acesso estruturado ao SDK
- [N/A] CSRF — aplicacao CLI, sem endpoints HTTP
- [N/A] Headers HTTP de seguranca — aplicacao CLI, sem servidor web

### Credenciais e configuracao
- [x] `.env` esta no `.gitignore` (linha 148)
- [x] `.env.agents` esta no `.gitignore` (linha 8)
- [x] `.env.example` contem apenas nomes de variaveis e valores default — nenhuma chave real
- [x] Nenhum arquivo `.env` rastreado pelo Git (verificado com `git ls-files`)
- [x] Nenhuma credencial hardcoded no codebase (verificado com grep por padroes de chave)
- [x] Todas as chaves API vem de variaveis de ambiente via `os.getenv()`
- [x] Dockerfile nao expoe segredos (sem `ENV`, `ARG` ou `EXPOSE` com chaves)

### SEC-03-001 — Fallback nao ecoa entrada do usuario
- [x] `AssistantCore._fallback_response()` — todas as mensagens sao textos genericos fixos, sem referencia a `user_text`, `cleaned` ou `transcript`
- [x] `voice_loop.py` — no caminho de erro STT (linhas 56-61), imprime `FALLBACK_ERROR_TEXT` (texto fixo), nao o transcript
- [x] `voice_loop.py` — nos caminhos de fallback TTS (linhas 74-93), imprime `response_text` (resposta do assistente, nao entrada do usuario) e `FALLBACK_POSITIVE_TEXT` (texto fixo)
- [x] `messages.py` — constantes `FALLBACK_POSITIVE` e `FALLBACK_ERROR` sao strings fixas sem interpolacao

### Remocao completa do Gemini
- [x] Zero referencias a `google-genai`, `google.generativeai`, `google-generativeai` ou `GOOGLE_API_KEY` no codebase ativo (unica ocorrencia esta no relatorio de seguranca antigo `docs/security/assistente-v1-etapa03-security.md`, que e historico)

### Script de geracao (generate_static_audio.py)
- [x] Chave ElevenLabs lida via `os.getenv()` — nunca impressa ou logada
- [x] Saida do script sao apenas arquivos WAV e mensagens de progresso sem dados sensiveis

### Dependencias
- [x] `groq==1.7.0` — pacote oficial da Groq no PyPI
- [x] `elevenlabs==2.75.0` — pacote oficial da ElevenLabs no PyPI
- [x] Nenhuma dependencia com vulnerabilidade critica conhecida identificada

---

## Vulnerabilidades criticas encontradas

**Nenhuma.**

---

## Observacoes nao-criticas (para comentario no PR)

### OBS-01: Resposta do LLM impressa no terminal em fallback TTS
- **Arquivo:** `app/assistant/voice_loop.py` linhas 76, 88, 93
- **Descricao:** Quando TTS falha ou nao esta configurado, `response_text` (saida do LLM) e impresso no terminal. Isto e comportamento esperado conforme o planejamento (usuario precisa ver a resposta quando audio falha). Porem, se o LLM for alvo de prompt injection, a resposta pode conter conteudo inesperado no terminal.
- **Risco:** Baixo — requer prompt injection bem-sucedido e acesso visual ao terminal.
- **Recomendacao:** Manter como esta. Monitorar em futuras features se houver inputs de fontes externas nao-confiaveies (hoje o input vem apenas do microfone do usuario).

### OBS-02: `sys.path.insert` no script de geracao
- **Arquivo:** `scripts/generate_static_audio.py` linha 100
- **Descricao:** O script manipula `sys.path` para importar `app.assistant.messages`. Padrao comum em scripts utilitarios, mas teoricamente permite import hijacking se o diretorio pai contiver modulo malicioso.
- **Risco:** Minimo — script executado apenas pelo desenvolvedor em ambiente local, nao em runtime.
- **Recomendacao:** Nenhuma acao necessaria.

---

## Conclusao

Auditoria concluida. **Sem vulnerabilidades criticas.** A feature T-001 esta aprovada do ponto de vista de seguranca. SEC-03-001 preservado em todos os caminhos de fallback. Chaves API protegidas por variaveis de ambiente sem exposicao em logs, codigo ou configuracao versionada. Duas observacoes nao-criticas registradas para comentario no PR quando disponivel.

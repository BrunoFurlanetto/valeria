# CLAUDE.md — valeria

> Configurações do projeto para o time de agentes.
> Complementa e sobrescreve o CLAUDE.md global quando necessário.

---

## Projeto

**Nome:** valeria
**Descrição:** Assistente virtual com detecção de wake-word via LSTM + Groq (chat+STT) + ElevenLabs (TTS)
**Stack:** Python 3.12+ · PyTorch · Groq · ElevenLabs · PyAudio
**Repositório:** https://github.com/BrunoFurlanetto/valeria.git
**Ambiente local:** `python -m app.assistant text` ou `python -m app.assistant run` (requer `GROQ_API_KEY` no `.env`)

---

## Vault (Obsidian)

**Path do projeto:** Dev projects/Valeria
**Memory:** Dev projects/Valeria/memory.md
**Planejamentos:** Dev projects/Valeria/features/

### Abertura de sessão

Ler o `memory.md` antes de qualquer task:

```
mcp_obsidian: read_note
path: Dev projects/Valeria/memory.md
```

Priorizar na leitura:
- Última entrada do `🕐 Log de Sessões` — o que estava em andamento e pendente
- ADRs relevantes à task atual
- Problemas conhecidos relacionados à feature

Se a task envolve planejamento de feature, ler também:
```
mcp_obsidian: read_note
path: Dev projects/Valeria/features/<nome-da-feature>.md
```

### Fechamento de sessão

Ao encerrar qualquer task, atualizar o `memory.md`:

```
mcp_obsidian: update_note
path: Dev projects/Valeria/memory.md
```

Seguir o protocolo de fechamento definido no CLAUDE.md global.

---

## Branches protegidas

Nunca commitar diretamente ou criar branches a partir de:
- `main`

O orchestrator sempre pergunta a branch base antes de criar qualquer branch nova.

---

## Como rodar

```bash
# Instalar dependências
pip install -r app/requirements.txt

# Assistente em modo texto
python -m app.assistant text

# Assistente em modo voz (com wake-word)
python -m app.assistant run --model-file app/network/save_model/valeria_optimized.zip

# Treinar modelo wake-word
python app/network/training.py \
  --train_data_json <train.json> \
  --test_data_json <test.json> \
  --save_checkpoint_path app/network/save_model \
  --model_name valeria_wake

# Otimizar modelo treinado
python app/network/optimized_model.py \
  --model_checkpoint app/network/save_model/valeria_wake.pt \
  --save_path app/network/save_model/valeria_optimized.zip

# Executar inferência wake-word
python app/network/exec.py \
  --model_file app/network/save_model/valeria_optimized.zip

# Regenerar áudios estáticos (earcons + frases de fallback)
python scripts/generate_static_audio.py
```

## Testes

```bash
python -m pytest -q
```

- Mudanças no modelo wake-word → rodar `training.py` em dataset reduzido
- Mudanças na inferência → rodar `exec.py` com modelo known-good
- Novos módulos → criar `tests/` com `pytest`

---

## Ownership dos agentes

| Agente | Pode criar/editar | Somente leitura |
|--------|------------------|-----------------|
| `dev` | `main.py`, `app/assistant/`, `scripts/`, `app/network/model.py`, `app/network/training.py`, `app/network/exec.py`, `app/network/optimized_model.py`, `app/utils/`, `.env.example`, `app/requirements.txt`, `CLAUDE.md`, `README.md`, `Dockerfile` | `app/network/dataset.py`, `app/network/save_model/` |
| `dba` | `app/network/dataset.py`, `app/network/save_model/` | todo o resto |
| `qa` | `tests/` | todo código de produção |
| `security` | `docs/security/` | todo o codebase |
| `reviewer` | — | todo o codebase |

---

## Convenções

### Commits
```
tipo(escopo): mensagem imperativa curta

Task: T-xxx
```
Tipos válidos: feat, fix, refactor, test, docs, chore, perf, style

### Python
- PEP 8: 4-space indent, snake_case funções/variáveis/arquivos, PascalCase classes
- CLI args via `argparse` para scripts reproduzíveis
- Módulos focados: pipeline de dados em `utils/`, modelo/runtime em `network/`

### Variáveis de ambiente obrigatórias
- `GROQ_API_KEY` — chave da API Groq (chat gpt-oss-20b + STT whisper-large-v3-turbo)
- `ELEVENLABS_API_KEY` — chave da API ElevenLabs (TTS eleven_flash_v2_5); opcional — sem ela usa fallback estático
- `ELEVENLABS_VOICE_ID` — ID da voz ElevenLabs; necessário se ELEVENLABS_API_KEY estiver presente

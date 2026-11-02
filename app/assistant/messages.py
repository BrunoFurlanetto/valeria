"""Constantes de texto para mensagens de fallback da Valeria.

Regra ADR-007: se o texto mudar aqui, re-executar scripts/generate_static_audio.py
e atualizar os WAVs correspondentes no mesmo commit.
"""

# Frase falada (via WAV estatico) quando chat funcionou mas TTS falhou
FALLBACK_POSITIVE = "Entendi seu pedido. A resposta esta no terminal."

# Frase falada (via WAV estatico) quando Groq STT ou chat falharam
FALLBACK_ERROR = "Desculpe, nao consegui processar seu pedido agora."

# Texto exibido no terminal nos mesmos cenarios
FALLBACK_POSITIVE_TEXT = FALLBACK_POSITIVE
FALLBACK_ERROR_TEXT = FALLBACK_ERROR

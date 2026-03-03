# GEOPOLITICAL EVENT ENCODING

Use mixed encoding:
1. Binary flags for event occurrence per day (simple, interpretable).
2. Severity score (0-1) = normalized function of:
   - count of event-related headlines that day
   - absolute sentiment magnitude (|FinBERT|)
   - third-party risk index (if available e.g., GDELT)
3. days_since_event for decay modeling (exponential decay factor)
4. Optionally: small text embedding (<= 64 dims) for rare complex events — only if you have stable signal

Guideline: use both binary + severity. This preserves interpretability and gives the model magnitude information.

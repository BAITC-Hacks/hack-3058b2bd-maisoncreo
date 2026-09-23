# hack-3058b2bd-maisoncreo
Hackathon team repository for Maisoncreo

## Optional live explanations

The optional OpenAI explanation layer validates every declared structured fact
against its canonical dataset value. This deliberately strict check does not
currently understand Russian inflection: for example, natural wording such as
`говорит по-русски` does not literally contain the canonical language token
`русский` and will therefore be rejected.

This is a known limitation of live explanations, not a recommendation failure.
When API output fails validation for this or any other reason, the application
automatically uses the deterministic, grounded fallback explanation. Contractor
selection and ordering are never changed by the explanation layer.

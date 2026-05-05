from typing import Any, Dict, List

# used to record preset questions, key is the target id
PRESET_QUESTION = "preset_question"

# redis keys
CAPTCHA_PREFIX = "cap_"
RSA_KEY = "rsa_"
USER_PASSWORD_ERROR = "user_password_error:{}"
USER_CURRENT_SESSION = "user_current_session:{}"
SESSION_CONTEXT_CACHE = "session_context_cache:{}"

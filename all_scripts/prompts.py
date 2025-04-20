def get_few_shot_prompt_old():

    FEW_SHOT_PROMPT = """
    Example 1:
    Commit Message: "Use version in SSL_METHOD not SSL structure.
    When deciding whether to use TLS 1.2 PRF and record hash algorithms use the version number in the corresponding SSL_METHOD structure instead of the SSL structure. The SSL structure version is sometimes inaccurate. Note: OpenSSL 1.0.2 and later effectively do this already. (CVE-2013-6449)"
    Function Code:
    " long ssl_get_algorithm2(SSL *s)
    {{
        long alg2 = s->s3->tmp.new_cipher->algorithm2;
        if (TLS1_get_version(s) >= TLS1_2_VERSION &&
            alg2 == (SSL_HANDSHAKE_MAC_DEFAULT|TLS1_PRF))
            return SSL_HANDSHAKE_MAC_SHA256 | TLS1_PRF_SHA256;
        return alg2;
    }}"
    Chain-of-Thought: The commit message clearly references CVE-2013-6449 and the function logic relies on an inaccurate version number from the SSL structure. This indicates a vulnerability.
    Answer: Yes
    
    Example 2:
    Commit Message: "None"
    Function Code:
    "gnutls_session_get_data (gnutls_session_t session,
                             void *session_data, size_t * session_data_size)
    {{
      gnutls_datum_t psession;
      int ret;
      if (session->internals.resumable == RESUME_FALSE)
        return GNUTLS_E_INVALID_SESSION;
      psession.data = session_data;
      ret = _gnutls_session_pack (session, &psession);
      if (ret < 0)
        {{
           gnutls_assert ();
           return ret;
         }}
      *session_data_size = psession.size;
      if (psession.size > *session_data_size)
         {{
           ret = GNUTLS_E_SHORT_MEMORY_BUFFER;
           goto error;
         }}
      if (session_data != NULL)
         memcpy (session_data, psession.data, psession.size);
      ret = 0;
    error:
      _gnutls_free_datum (&psession);
      return ret;
    }}"
    Chain-of-Thought: The function calls memcpy without sufficient boundary checks and the context mentions CWE-119 and CVE-2011-4128. This is indicative of a buffer overflow vulnerability.
    Answer: Yes
    
    Example 3:
    Commit Message: "None"
    Function Code:
    "gnutls_session_get_data (gnutls_session_t session,
                             void *session_data, size_t * session_data_size)
    {{
      gnutls_datum_t psession;
      int ret;
      if (session->internals.resumable == RESUME_FALSE)
        return GNUTLS_E_INVALID_SESSION;
      psession.data = session_data;
      ret = _gnutls_session_pack (session, &psession);
      if (ret < 0)
        {{
          gnutls_assert ();
          return ret;
        }}
      if (psession.size > *session_data_size)
         {{
           ret = GNUTLS_E_SHORT_MEMORY_BUFFER;
           goto error;
         }}
      if (session_data != NULL)
        memcpy (session_data, psession.data, psession.size);
      ret = 0;
    error:
      _gnutls_free_datum (&psession);
      return ret;
    }}"
    Chain-of-Thought: This function is nearly identical to Example 2. It exhibits a risk of buffer overflow due to inadequate checking before memcpy. The associated vulnerability is confirmed by CWE-119.
    Answer: Yes
    
    Now, I want you to act as a software vulnerability detector. Based on the commit message and the code snippet provided, determine if the commit introduces a vulnerability. Answer with "Yes" for vulnerable and "No" for not vulnerable. Provide a brief explanation of your reasoning. 
    
    Commit Message: "{commit_message}"
    Function Code:
    {func}
    Answer: 
     
    """
    return FEW_SHOT_PROMPT
    


def get_zero_shot_prompt():

    ZERO_SHOT_PROMPT = """
    You are a software vulnerability detector. Based on the commit message and the code snippet provided, determine if the function introduces a vulnerability. You MUST answer with "Yes" for vulnerable and "No" for not vulnerable. Provide a brief explanation of your reasoning. 
    
    Commit Message: "{commit_message}"
    Function Code:
    {func}
    Answer: 
     
    """
    return ZERO_SHOT_PROMPT
    


# def get_think_llm_as_judge_eval_plan_prompt():

#     PROMPT = """
#     We want to evaluate the quality of the responses provided by AI assistants to the user question displayed
# below. For that, your task is to help us build an evaluation plan that can then be executed to assess the response quality. Whenever appropriate, you can choose to also include a step-by-step reference answer as part of the
# evaluation plan. Enclose your evaluation plan between the tags "[Start of Evaluation Plan]” and “[End of Evaluation Plan]".

# [User Instruction]
# Commit Message: {commit_message}
# Function Code: {func}
    
# [Question]You are a software vulnerability detector. Based on the commit message and the code snippet provided above, determine if the function introduces a vulnerability. You MUST answer with "Yes" for vulnerable and "No" for not vulnerable. Provide a brief explanation of your reasoning. 

#     """

#     return PROMPT


def get_think_llm_as_judge_eval_plan_prompt():
    
    return [
        {
            "role": "system",
            "content": (
                "We want to evaluate the quality of the responses provided by AI assistants to the user question displayed below. "
                "For that, your task is to help us build an evaluation plan that can then be executed to assess the response quality. "
                "Whenever appropriate, you can choose to also include a step-by-step reference answer as part of the evaluation plan. "
                "Enclose your evaluation plan between the tags \"[Start of Evaluation Plan]\" and \"[End of Evaluation Plan]\"."
            ),
        },
        {
            "role": "user",
            "content": (
                "[User Instruction]\n"
                "Commit Message: {commit_message}\n"
                "Function Code: {func}\n\n"
                "[Question] You are a software vulnerability detector. Based on the commit message and the code snippet provided above, determine if the function introduces a vulnerability. "
                "You MUST answer with \"Yes\" for vulnerable and \"No\" for not vulnerable. Provide a brief explanation of your reasoning.\n\n"
            ),
        },
    ]



EVAL_PLAN_FOR_COMPARISON = """
To evaluate the response quality of AI assistants in determining whether the provided function introduces a vulnerability, we will follow these steps:

1. **Verification of Obvious or Subtle Errors**: Check if the response contains any obvious errors, such as incorrect terminology or factual mistakes. Also, look for subtle errors, such as misunderstandings of the code snippet or the context.

2. **Evaluation Rubric**: Establish a clear evaluation rubric with specific criteria to assess the response quality. For this question, the criteria are:
   - **Accuracy**: Does the response correctly identify whether the function introduces a vulnerability?
   - **Explanation**: Is the provided explanation clear, concise, and relevant to the code snippet and the context?
   - **Relevance**: Is the response focused on the specific question asked, without introducing unnecessary information?
   - **Technical Correctness**: Does the response demonstrate a good understanding of the technical concepts involved, such as memory management and compiler destruction?

3. **Response Analysis**: Analyze each response carefully along the established criteria. For the "Accuracy" criterion, check if the response correctly identifies whether the function introduces a vulnerability. For the "Explanation" criterion, evaluate the clarity and relevance of the provided explanation. For the "Relevance" criterion, check if the response is focused on the specific question asked. For the "Technical Correctness" criterion, assess the response's demonstration of technical understanding.

4. **Evaluation of Responses**: Based on the analysis, evaluate each response's overall quality. Assign a score or ranking to each response, reflecting its performance across the evaluation criteria.

5. **Final Verdict**: Determine the best response based on the evaluation. The response that scores the highest or is ranked the best overall is considered the highest-quality response.
"""

def get_think_llm_as_judge_exec_plan_prompt():

    # PROMPT = """
    # Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user
    # question displayed below. Your evaluation should be performed by following the provided evaluation plan step-by-step.
    # Avoid copying the plan when doing the evaluation. Please also only stick to the given plan and provide explanation of
    # how the plan is executed to compare the two responses. Avoid any position biases and ensure that the order in which
    # the responses were presented does not influence your decision. Do not allow the length of the responses to influence
    # your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your evaluation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.
    
    # [User Question]
    # Commit Message: {commit_message}
    # Function Code: {func}
    # You are a software vulnerability detector. Based on the commit message and the code snippet provided above, determine if the function introduces a vulnerability. You MUST answer with "Yes" for vulnerable and "No" for not vulnerable. Provide a brief explanation of your reasoning. 
    
    # [The Start of Assistant A’s Answer]
    # {response_a}
    # [The End of Assistant A’s Answer]
    # [The Start of Assistant B’s Answer]
    # {response_b}
    # [The End of Assistant B’s Answer]
    # [The Start of Evaluation Plan]
    # {eval_plan}
    # [The End of Evaluation Plan]
    
    # """

    # return PROMPT

    return [
        {
            "role": "system",
            "content": (
                "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
                "You should choose the assistant that follows the user's instructions and answers the user's question better. "
                "Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Give accuracy criteria the highest priority. "
                "Begin your evaluation by comparing the two responses and provide a short explanation. "
                "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "
                "Do not allow the length of the responses to influence your evaluation. "
                "Do not favor certain names of the assistants. Be as objective as possible. "
                "After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better. "
            ),
        },
        {
            "role": "user",
            "content": (
                "[User Instruction]\n"
                "Commit Message: {commit_message}\n"
                "Function Code: {func}\n\n"
                "[Question] You are a software vulnerability detector. Based on the commit message and the code snippet provided above, determine if the function introduces a vulnerability.\n"
                "You MUST answer with \"Yes\" for vulnerable and \"No\" for not vulnerable. Provide a brief explanation of your reasoning.\n\n"
                "[The Start of Assistant A's Answer]\n"
                "{response_a}\n"
                "[The End of Assistant A's Answer]\n\n"
                "[The Start of Assistant B's Answer]\n"
                "{response_b}\n"
                "[The End of Assistant B's Answer]\n\n"
                "[The Start of Evaluation Plan]\n"
                "{eval_plan}\n"
                "[The End of Evaluation Plan]\n"
            ),
        },
    ]    
    

def get_think_llm_as_judge_verdict_prompt():

    PROMPT = """
    Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the
    user question displayed below. You should choose the assistant that follows the user’s instructions and answers
    the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy,
    depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses
    and provide a short explanation. Avoid any position biases and ensure that the order in which the responses
    were presented does not influence your decision. Do not allow the length of the responses to influence your
    evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.
    [[User Question]]
    {instruction}
    [The Start of Assistant A’s Answer]
    {response A}
    [The End of Assistant A’s Answer]
    [The Start of Assistant B’s Answer]
    {response B}
    [The End of Assistant B’s Answer]
    
    """

    return PROMPT


def get_think_llm_as_judge_complete_prompt():

    PROMPT = """
    User Instruction: 
    Commit Message: {commit_message}
    Function Code: {func}
    
[Question]You are a software vulnerability detector. Based on the commit message and the code snippet provided above, determine if the function introduces a vulnerability. You MUST answer with "Yes" for vulnerable and "No" for not vulnerable. Provide a brief explanation of your reasoning. 
    [The Start of Assistant A’s Answer]
    {response A}
    [The End of Assistant A’s Answer]

    [The Start of Assistant B’s Answer]
    {response B}
    [The End of Assistant B’s Answer]

[Start of Evaluation Plan]
{evaluation_plan}
[End of Evaluation Plan]
[Start of Plan Execution]
{exec_plan}
[End of Plan Execution]

    """
    
    





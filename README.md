# model inference 套件
* first edit: 7/20
* second edit: 8/24 -- 開發完ollama接口

## 使用方式：
* conda create a virtual environment
* pip install -r requirements
* write .env file
    - AZURE_OPENAI_API_KEY=***
    - AZURE_OPENAI_ENDPOINT=***
* initial LanguageModelHandler
    - import module
    ```
    from langchain_model_inference import LanguageModelHandler  # 待定
    ```
    - 檢視目前現有之模型(未來可自行新增)：
        - gpt-4
        - gpt-35-turbo
        - llama3.1
    - 實例化物件：
    ```
    system_prompt_test = [
        "The sentence you are given might be too wordy, complicated, or unclear. Rewrite the sentence and make your writing clearer by keeping it concise. Whenever possible, break complex sentences into multiple sentences and eliminate unnecessary words.",
        "Analyze the word choice, phrasing, punctuation, and capitalization in the given email. How may the writer of this email sound to the reader? These tones include Disheartening, Accusatory, Worried, Curious, Surprised, Disapproving, Unassuming, Formal, Assertive, Confident, Appreciative, Concerned, Sad, Informal, Regretful, Encouraging, Egocentric, Joyful, Optimistic, and Excited."
    ]
    user_question_test = [
        "If you have any questions about my rate or if you find it necessary to increase or decrease the scope for this project, please let me know.",
        "Hi Jen, I hope you're well. Can we catch up today? I'd appreciate your input on my presentation for tomorrow's meeting. I'd especially love it if you could double-check the sales numbers with me. There's a coffee in it for you!"
    ]
    test_model = LanguageModelHandler(
        model_name="gpt-35-turbo",
        user_question=user_question_test,
        system_prompt=system_prompt_test
        )

    test_model.get_model_inference()
    ```
    
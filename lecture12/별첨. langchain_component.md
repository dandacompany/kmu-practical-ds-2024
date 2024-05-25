## LangChain 컴포넌트 개념과 에시

### 1. 스키마

- **텍스트**: 텍스트 인터페이스로 언어모델과 주로 상호작용.

```python
from langchain.schema import Text

text_input = Text(content="Hello, how can I assist you today?")

```

- **채팅 메시지**: 시스템, 사용자, AI의 채팅 메시지.

```python
from langchain.schema import SystemChatMessage, HumanChatMessage, AIChatMessage

system_message = SystemChatMessage(content="You are an AI assistant.")
user_message = HumanChatMessage(content="What's the weather like today?")
ai_message = AIChatMessage(content="The weather is sunny with a slight chance of rain.")

```

- **예시**: 모델 훈련 및 평가용 입력/출력 쌍.

```python
from langchain.schema import Example

example = Example(input="Translate 'Hello' to Spanish", output="Hola")

```

- **문서**: 비구조화된 데이터 조각.

```python
from langchain.schema import Document

document = Document(page_content="This is a sample document content.", metadata={"author": "John Doe"})

```

### 2. 모델

- **대규모언어모델(LLMs)**: 대규모 언어 모델.

```python
from langchain.llms import OpenAI

llm = OpenAI(api_key="your-api-key")
response = llm.predict("What is the capital of France?")

```

- **채팅 모델**: 채팅 형식의 모델.

```python
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temporature=0)
chat_response = chat_model.chat([user_message])

```

- **텍스트 임베딩 모델**: 텍스트 임베딩을 생성하는 모델.

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_text("Sample text to embed.")

```

### 3. 프롬프트

- **프롬프트 값(PromptValue)**: 모델에 입력되는 값.

```python
from langchain.prompts import PromptValue

prompt_value = PromptValue(text="Tell me a joke.")

```

- **프롬프트 템플릿**: 다양한 입력 구성을 지원하는 템플릿.

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(input_variables=["location"], template="What is the weather like in {location}?")
prompt = template.format(location="New York")

```

- **출력 파서**: 모델 출력 형식을 지정하고 파싱.

```python
from langchain.output_parsers import OutputParser

class SimpleOutputParser(OutputParser):
    def parse(self, response: str):
        return response.strip()

parser = SimpleOutputParser()
parsed_output = parser.parse("  Here is your result.  ")

```

### 4. 인덱스

- **문서 로더**: 다양한 파일 형식을 텍스트 데이터로 변환.

```python
from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(file_path="sample.pdf")
documents = loader.load()

```

- **텍스트 분할기**: 긴 텍스트를 여러 조각으로 나눔.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=100)
chunks = splitter.split("This is a long text that needs to be split into smaller parts.")

```

- **벡터 저장소**: 임베딩된 벡터를 저장하고 검색.

```python
from langchain.vectorstores import FAISS

vector_store = FAISS()
vector_store.add_texts(["This is a sample document.", "Another document for testing."])

```

- **검색기**: 쿼리를 입력으로 받아 관련 문서 반환.

```python
from langchain.retrievers import SimpleRetriever

retriever = SimpleRetriever(vector_store=vector_store)
results = retriever.retrieve("sample query")

```

### 5. 체인

- **체인(Chain)**: 여러 구성 요소를 결합하여 특정 목적 수행.

```python
from langchain.chains import SimpleChain

chain = SimpleChain(prompt_template=template, model=llm, output_parser=parser)
result = chain.run(input_variables={"location": "Paris"})

```

- **LLM 체인**: 프롬프트 템플릿과 모델을 조합해 사용.

```python
from langchain.chains import LLMChain

llm_chain = LLMChain(prompt_template=template, model=llm)
response = llm_chain.run({"location": "London"})

```

- **인덱스 연관 체인**: 문서와 상호작용하여 질문에 답변.

```python
from langchain.chains import IndexRelatedChain

index_chain = IndexRelatedChain(index=vector_store, model=llm)
answer = index_chain.run(query="What is the document about?")

```

### 6. 메모리

- **채팅 메시지 히스토리**: 이전 채팅 상호작용을 기억.

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_message(user_message)
history.add_message(ai_message)

```

- **장기 및 단기 메모리**: 대화 데이터 저장 및 검색.

```python
from langchain.memory import LongTermMemory, ShortTermMemory

long_term_memory = LongTermMemory()
short_term_memory = ShortTermMemory()

long_term_memory.add("User asked about weather.")
short_term_memory.add("Assistant replied it's sunny.")

```

### 7. 에이전트

- **도구(Tools)**: 언어 모델이 리소스와 상호작용.

```python
from langchain.tools import WebSearchTool

web_search_tool = WebSearchTool(api_key="your-api-key")
search_results = web_search_tool.query("Latest news about AI")

```

- **툴킷(Toolkits)**: 특정 작업을 수행하는 도구 세트.

```python
from langchain.toolkits import DataAnalysisToolkit

toolkit = DataAnalysisToolkit(tools=[web_search_tool])
analysis_results = toolkit.analyze("Analyze data trends in AI research.")

```

- **에이전트(Agents)**: 사용자 입력을 받아 액션을 수행.

```python
from langchain.agents import BasicAgent

agent = BasicAgent(model=llm, tools=[web_search_tool])
agent_response = agent.handle_input("Find recent articles about machine learning.")

```

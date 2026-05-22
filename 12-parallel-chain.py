from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGroq(model='openai/gpt-oss-120b')
model2 = ChatGroq(model='llama-3.3-70b-versatile')

promt1 = PromptTemplate(
    input_variables=['text'],
    template="generate simple notes from the following text \n {text}"
)
promt2 = PromptTemplate(
    input_variables=['text'],
    template="generate a quiz containing 5 questions from the following text \n {text}"
)
promt3 = PromptTemplate(
    input_variables=['notes','quiz'],
    template="mergt the following notes and quiz into a single document \n notes->{notes}, quiz->{quiz}"
)

parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'notes': promt1 | model1 | parser,
    'quiz' : promt2 | model2 | parser
}) 

marg_chain = promt3 | model1 | parser

final_chain = parallel_chain | marg_chain

doc = r"""
The Transformer in deep learning is a neural network architecture that relies entirely on a mechanism called Self-Attention to compute representations of its input and output without using sequential processing (like RNNs) or local processing (like CNNs).Introduced in the 2017 landmark paper "Attention Is All You Need" by Vaswani et al., it serves as the foundational architecture for modern Large Language Models (LLMs) like GPT-4, Gemini, and Claude.Why the Transformer Was InventedBefore Transformers, models like RNNs (Recurrent Neural Networks) and LSTMs processed text sequentially—one word at a time. This caused two major bottlenecks:No Parallelization: You cannot compute the 10th word until you have computed the 9th word. This made training on massive datasets incredibly slow.Vanishing Gradients / Long-Range Forgetting: RNNs struggle to retain context when words are far apart in a long paragraph.The Transformer solved both issues. It processes all words in a sequence simultaneously (in parallel) and allows every word to look at every other word directly, regardless of distance.Core Components & ArchitectureA standard Transformer uses an Encoder-Decoder architecture, though modern models can be Encoder-only (e.g., BERT) or Decoder-only (e.g., GPT).Input Tokens ──> [Input Embedding + Positional Encoding] ──> [Encoder Layers] ──> Context Vectors -> Output Tokens ─> [Output Embedding + Positional Encoding] ─> [Decoder Layers] 
                                                                     │
                                                             [Linear & Softmax] ──> Next Word Prediction
1. Input Embedding & Positional EncodingInput Embedding: Converts raw words into dense vectors of numbers (e.g., a vector of size 512) that capture semantic meaning.Positional Encoding: Because the Transformer processes all words simultaneously, it has no inherent sense of word order. To fix this, static sine and cosine wave values (or learned vectors) are mathematically added to the word embeddings to inject the precise position of each word in the sentence.2. The Self-Attention Mechanism (The Core Engine)Self-attention allows the model to look at other words in the input sentence to better understand the current word. For example, in "The animal didn't cross the street because it was too tired", self-attention connects "it" strongly to "animal".To calculate Self-Attention, the embedding of each word is transformed into three vectors via learned linear projections:Query (\(Q\)): "What am I looking for?" (The current word asking a question).Key (\(K\)): "What do I contain?" (Other words offering their profile).Value (\(V\)): "What information do I actually hold?" (The actual content to be extracted).The mathematical formula for Scaled Dot-Product Attention is:\(\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V\)\(QK^{T}\) computes the similarity score between every Query and every Key.Dividing by \(\sqrt{d_{k}}\) (the dimension of the keys) scales down the numbers to prevent vanishing gradients during training.softmax converts these scores into probabilities (attention weights) that add up to 1.Multiplying by \(V\) extracts a weighted sum of the values, emphasizing the words that matter most.3. Multi-Head AttentionInstead of performing self-attention once, the model splits \(Q, K,\) and \(V\) into multiple lower-dimensional "heads" (e.g., 8 heads) and computes attention in parallel.One head might focus on grammatical relationships (verb to object).Another head might focus on pronouns (linking "it" to "animal").The outputs of all heads are concatenated and linearly projected back to the original size.4. Feed-Forward Networks (FFN)After the Multi-Head Attention layer, the output vector of each word passes through a standard fully connected feed-forward network. This network operates on each word position independently and identically, applying non-linear transformations to extract deeper features.5. Residual Connections & Layer NormalizationTo train incredibly deep networks (e.g., 24 to 100+ layers) without performance degrading:Residual Connections (Skip Connections): The original input to a layer is added directly to its output (\(x + \text{Layer}(x)\)). This allows gradients to flow backwards through the network smoothly during training.Layer Normalization (LayerNorm): Stabilizes the training process by normalizing the inputs across features for each individual training sample.The Encoder vs. The DecoderFeatureThe EncoderThe DecoderPrimary GoalUnderstand and extract features from the source text.Generate new text auto-regressively (one token at a time).Attention TypeBidirectional Self-Attention: Can look at both future and past words simultaneously.Masked Self-Attention: Can only look at past words. Future words are hidden (masked) so it can't cheat.Cross-AttentionNone.Encoder-Decoder Attention: Its Queries look at the Encoder's Keys and Values to pull context from the input text.ExamplesBERT, RoBERTaGPT series, LLaMA, MistralWhy Transformers Dominate Deep LearningMassive Scale: They scale predictably. If you increase data, compute, and model parameters, the model consistently gets smarter (known as Scaling Laws).Cross-Modal Versatility: The architecture is not limited to text. By breaking images into patches, you get Vision Transformers (ViT). By slicing audio waveforms, you get advanced speech recognition models like Whisper.Transfer Learning: Models can be pre-trained on the entire internet using self-supervised learning (predicting missing words) and then fine-tuned on highly specific tasks with very little data.
"""

result = final_chain.invoke({'text':doc})

print(result)

final_chain.get_graph().print_ascii()
"""
Text splitters and chunking strategies
Optimizing document chunks for RAG
"""

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SIMPLE_TEXT = """
# Introduction to Machine Learning

Machine learning is a transformative field that enables computers to learn patterns from data and make decisions without explicit programming. This guide introduces the fundamental concepts, types, and real-world applications of machine learning.

## What is Artificial Intelligence?

**Artificial intelligence (AI)** refers to computer systems designed to perform tasks that typically require human intelligence. AI encompasses a broad range of techniques, from simple rule-based systems to sophisticated neural networks. The key advantage of AI is its ability to process vast amounts of data and identify patterns that would be impractical for humans to detect manually.

## What is Machine Learning?

**Machine learning (ML)** is a subset of artificial intelligence that focuses on enabling systems to learn from data automatically. Rather than following pre-programmed instructions, machine learning algorithms improve their performance through experience. As they encounter more data, they refine their understanding and make better predictions or decisions.

## Types of Machine Learning

Machine learning encompasses several distinct approaches, each suited to different problem types:

- **Supervised Learning**: The algorithm learns from labeled data (input-output pairs). Common tasks include classification and regression. Examples: email spam detection, house price prediction.

- **Unsupervised Learning**: The algorithm finds hidden patterns in unlabeled data without predefined outcomes. Common tasks include clustering and dimensionality reduction. Examples: customer segmentation, anomaly detection.

- **Reinforcement Learning**: The algorithm learns by interacting with an environment, receiving rewards or penalties for actions. Examples: game AI, autonomous robotics, trading systems.

- **Deep Learning**: A specialized subset using neural networks with multiple layers to process complex data like images and text. Examples: image recognition, natural language processing.

## Real-World Applications of Machine Learning

Machine learning drives innovation across industries:

| Application | Use Case |
|---|---|
| **Predictive Analytics** | Forecasting sales trends, customer churn, equipment failures |
| **Natural Language Processing** | Chatbots, sentiment analysis, machine translation, text summarization |
| **Computer Vision** | Face recognition, object detection, medical image analysis |
| **Recommendation Systems** | Personalized product suggestions, content recommendations |
| **Healthcare** | Disease diagnosis, drug discovery, personalized treatment plans |

These applications demonstrate how machine learning solves real problems by automating complex tasks and uncovering insights from data.

"""

SIMPLE_CODE = '''
# Python Machine Learning Code Snippet

## Class: SimpleLinearRegression

A basic implementation of linear regression from scratch.

```python
class SimpleLinearRegression:
    """A simple linear regression model using gradient descent."""
    
    def __init__(self, learning\\_rate=0.01, iterations=1000):
        """
        Initialize the model.
        
        Args:
            learning\\_rate (float): Controls the step size for gradient descent.
            iterations (int): Number of training iterations.
        """
        self.learning\\_rate = learning\\_rate
        self.iterations = iterations
        self.slope = 0
        self.intercept = 0
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Args:
            X (list or array): Input features.
            y (list or array): Target values.
        """
        n = len(X)
        
        for _ in range(self.iterations):
            # Predictions
            y\\_pred = self.slope * X + self.intercept
            
            # Calculate gradients
            slope\\_gradient = (-2 / n) * sum(X * (y - y\\_pred))
            intercept\\_gradient = (-2 / n) * sum(y - y\\_pred)
            
            # Update parameters
            self.slope -= self.learning\\_rate * slope\\_gradient
            self.intercept -= self.learning\\_rate * intercept\\_gradient
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (list or array): Input features.
            
        Returns:
            Predicted values.
        """
        return self.slope * X + self.intercept

'''


def recursive_splitter():
    """You can use separators to guide how it should split the text.
    Overlap is important so it can include overlapping content from previous
    and next chunks. This will allow retrival to get a more completed answer than
    chunks that don't have any connection between them. It can lead to incomplete answers or
    not finding relevant answer at all.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text=SIMPLE_TEXT)
    print(f"Original Text Length: {len(SIMPLE_TEXT)}")
    print(f"Number of Chunks: {len(chunks)}")
    print(f"Chunk Sizes: {[len(c) for c in chunks]}")
    print(f"First Chunk: {chunks[0]}")


def overlap_importance():
    text = "This is a sample text. This is another sentence." * 100

    no_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=0
    )
    with_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20
    )

    chunks_no_overlap = no_overlap_splitter.split_text(text)
    chunks_with_overlap = with_overlap_splitter.split_text(text)

    print("Without Overlap:")
    print(f"Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f"Chunk 2 start: {chunks_no_overlap[1][:20]}...")

    print("With Overlap:")
    print(f"Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
    print(f"Chunk 2 start: {chunks_with_overlap[1][:20]}...")


if __name__ == "__main__":
    # recursive_splitter()
    overlap_importance()

# Information Retrieval System for a Video Game Website

## Overview

This Python-based Information Retrieval (IR) system is designed for indexing and searching content related to a specialized video game website. However, it's built to be domain-independent, allowing adaptability to other contexts. The system offers various experimentation options, including text preprocessing techniques (e.g., lemmatization, stemming), query expansion, spell checking, and indexing optimization methods.

## System Outline

### Command Line Options

The system supports several command line options:

- `-v`: Provides verbose output for users, including logging and function timing.
- `--searcher`: Allows selection between using the vector space model or a naive total token score method for searching.
- `-r`: Reindexes the documents.
- Other flags for enhancing search results and experimentation and can be found by running `python main.py -h`

### Indexing Methods

Users can choose indexing methods:

- `async`, `multithreaded`, `multi-processed`, or `synchronous`.
- Experimentation revealed multiprocessing as the fastest, suggesting processing power, not IO, as the bottleneck.

#### Parse Time Comparison (Table 1)

| Parser | Average Time |
| ------ | ------------ |
| Sync   | 30.7s        |
| Async  | 29.7s        |
| MP     | 24.2s        |
| MT     | 57.6s        |

### Parsing and Inverted Index

- Parsing generates a modified Inverted Index, pickled for quicker startup times.
- On startup or after parsing, the Inverted Index creates a vector space and document matrix for vector space searching.

#### Modified Inverted Index Types (Figure 2)

The modified Inverted Index includes pre-calculated term frequency, IDF, and other data. This optimization minimizes recalculation during search operations, enhancing efficiency.

## Dependencies and Usage

- The system is written in Python 3.11 and primarily utilizes NLTK for text processing.

### Setup and Execution

1. Install Python 3.11 and NLTK.
2. Clone the repository and navigate to the project directory.
3. Run the system using command-line options as needed:
   ```bash
   python main.py -v --searcher vector_space -r --indexing-method multi-processed
   ```

### Experimentation and Customization

- Users can experiment with different flags, indexing methods, and preprocessing techniques to optimize performance and search results.

## Contributors

- Jacob Edwards

## License

- Unlicensed Project

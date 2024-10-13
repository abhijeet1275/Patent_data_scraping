# Patent Search Engine

## Overview

This project implements a Patent Search Engine that utilizes both BM25 for keyword search and semantic search using embeddings. The goal is to efficiently retrieve relevant patents from a dataset of approximately 1000 patent JSON files, each containing various attributes, including abstracts.

## Features

- **Keyword Search**: Implemented using the BM25 algorithm, allowing users to perform efficient text-based searches on patent abstracts.
- **Semantic Search**: Utilizes SentenceTransformers to compute embeddings for patent abstracts, enabling semantic matching and retrieval based on the meaning of the search queries.
- **User-Friendly Interface**: Built using Streamlit, providing an interactive web application for users to perform searches and view results.


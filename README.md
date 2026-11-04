# Product Matching

A data science take-home challenge (2022): match product listings from an e-commerce marketplace using only the text that sellers enter when they register a product.

- [`challenge_v1.ipynb`](challenge_v1.ipynb): the solution notebook.
- [`base_info_produtos.csv`](base_info_produtos.csv): the product sample (26,241 listings, tab-separated). The notebook reads this file.
- [`requirements.txt`](requirements.txt): the conda environment used to run the notebook.

## The challenge

*Translated from the original Portuguese.*

### Context

One of the biggest challenges for marketplace e-commerce platforms is comparing product information based on text descriptions. Each seller registered on the platform can enter product data in very different ways, sometimes with incomplete information, and even the most basic description, the name shown on the product page, is not standardized.

Grouping identical products matters for everything from pricing strategy and sales forecasting to search and recommendation. Duplicate products in the catalog also make inventory management harder and marketing strategies less efficient.

### Problem

Magazine Luiza, one of the largest retailers in Brazil, has a large product catalog, much of it coming from external sellers on its marketplace. The sample in the CSV file contains some of the information sellers provided when registering each product: the product's **name**, **type**, **category**, **brand**, **color** and **model**. The file is tab-separated (`\t`), and fields that were not filled in contain the word `None`.

Build a matching solution using the data provided. The solution should answer the following questions:

1. **(a)** Given two products, what is the probability that they are the same product?
2. **(b)** Given a product name, how can the type, category, brand, color and model be recovered? If the product does not have a piece of information, the system should return `None`.
3. **(c)** How can the matching solution be used to return the most relevant products for a customer's simple search on the website? Example search: *notebook sony 4 GB*.
   Note: adapt the matching solution you built to solve this item; there is no need to build a system or an API.

### Evaluation criteria

- Understanding of the problem and solution design.
- Knowledge of natural language processing and machine learning models.
- Ability to explain complex ideas simply, that is, to communicate with people who don't share the same academic background or understanding of technology.

## Data columns

The CSV columns keep their original Portuguese names:

| Column | Meaning |
|---|---|
| `nome` | Product name |
| `tipo` | Type |
| `marca` | Brand |
| `categoria` | Category |
| `cor` | Color |
| `modelo` | Model |

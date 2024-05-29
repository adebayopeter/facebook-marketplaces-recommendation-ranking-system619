# Facebook Marketplace's Recommendation Ranking System

Facebook marketplace recommendation ranking system project 

## Table of Contents

1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File Structure](#file-structure)
5. [License](#license)

## Description

The Facebook Marketplace is a platform for buying and selling products on Facebook. This project is an implementation of the system behind the marketplace, which uses AI to recommend the most relevant listings based on a personalised search query.

## Installation

To install the Facebook Marketplace project, you can clone the repository from GitHub:

```bash
git clone https://github.com/adebayopeter/facebook-marketplaces-recommendation-ranking-system619.git
```
Navigate to the project directory and install the required dependencies:
```
cd facebook_marketplace
pip install -r requirements.txt
```
Create a folder `data` that would serve as our main resource folder.

Inside `data` create a folder `csv` that holds the csv dataset of `Image.csv`, `Prodcts.csv`, `merged_data.csv` and `cleaned_images.csv`. 

Inside the folder, create `images` folder that holds the image dataset and `clean_images` folder where we store the cleaned images.

## File Structure
The project directory is structured as follows:

```
📦 facebook_markeplace
├─ data
│  ├─ csv
│  │  ├─ Products.csv
│  │  ├─ training_data.csv
│  │  ├─ merged_data.csv
│  │  └─ cleaned_images.csv
│  ├─ clean_images
│  └─ images
├─ clean_images.py
├─ clean_tabular_data.py
├─ main.py
├─ README.md
├─ requirements.txt
└─ .gitignore
```
## License
This project is licensed under [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
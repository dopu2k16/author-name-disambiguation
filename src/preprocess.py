import os
import xml.etree.ElementTree as ET
import re
from typing import Optional

import pandas as pd

from nltk.corpus import stopwords
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

cited_authors_dict = {}

PARSING_ERROR = 'Parsing Error, check parsing step'


def extract_article_title_xml(root: ET.Element):
    """
    Extract title from a single lxml article file
    Task 1:
    :param root:
    :return: str: title of a single article
    """
    try:
        title_element = root.find(".//ArticleTitle")
        if title_element is not None and title_element.itertext() is not None:
            return "".join(title_element.itertext()).strip()
        else:
            return None
    except ET.ParseError as e:
        print(PARSING_ERROR, e)


def extract_all_articles(dir_path: str):
    """
    Task 1: Extract title of all articles
    :param dir_path:
    :return:
    """
    titles = []
    try:
        for filename in os.listdir(dir_path):
            if not filename.startswith('.') and filename.endswith('.xml'):
                parser = ET.XMLParser(encoding="utf-8")
                tree = ET.parse(os.path.join(dir_path, filename), parser=parser)
                root = tree.getroot()
                title = extract_article_title_xml(root)
                titles.append(title)
    except FileNotFoundError:
        print("Wrong Directory Path")
    except ET.ParseError:
        print("Parsing Error, check parsing step")
    return titles


def print_sort_alphabetic(arr: list, item_type: str):
    arr.sort()
    for ind, item in enumerate(arr):
        print(f"{item_type}: {ind + 1}", item)


def extract_all_articles_authors(dir_path: str):
    """
    Task 2 Extracting the list of authors of all articles
    :param dir_path:
    :return:
    """
    authors = []
    try:
        for filename in os.listdir(dir_path):
            if not filename.startswith('.') and filename.endswith('.xml'):
                tree = ET.parse(os.path.join(dir_path, filename))
                root = tree.getroot()
                # Find the author elements and append their names to the list of authors
                author_elements = root.findall(".//Author")
                for author_element in author_elements:
                    name_elements = author_element.findall("./AuthorName/*")
                    name = ''
                    for element in name_elements:
                        if element.text is not None:
                            name += element.text.strip() + ' '
                    if name.strip() != '':
                        authors.append(name.strip())
    except FileNotFoundError as e:
        print("Wrong Directory Path")
    except ET.ParseError as e:
        print(PARSING_ERROR, e)
    return authors


def extract_bibliography_of_article(root: ET.Element):
    """
    Task 3
    Extracting all author names of references and citation count of ref authors
    :param root:
    :return: author_list_total, cited_authors_dict
    """
    citation_elements = root.findall(".//Citation")
    # print(filename)
    authors_list_total = []

    # Loop through the citation elements and add the authors to the cited authors dictionary
    for citation_element in citation_elements:
        # Find all the author elements in the citation
        author_elements = citation_element.findall(".//BibAuthorName")
        if len(author_elements) > 0:
            # Get the author names and add them to the cited authors dictionary
            author_names = []
            for author_element in author_elements:
                name_elements = author_element.findall(".//Initials") + author_element.findall(".//FamilyName")
                name = ' '.join(
                    [name_element.text for name_element in name_elements if
                     name_element.text is not None]).strip()
                # print(name)
                if name != '':
                    author_names.append(name)
                    # print(author_names)
            if len(author_names) > 0:
                authors_list_total.append(author_names)
                # Add the author names to the cited authors dictionary
                for author_name in author_names:
                    if author_name not in cited_authors_dict:
                        cited_authors_dict[author_name] = 1
                    else:
                        cited_authors_dict[author_name] += 1

    return authors_list_total, cited_authors_dict


def extract_ref_authors(dir_path: str):
    ref_authors = []
    try:
        for filename in os.listdir(dir_path):
            if not filename.startswith('.') and filename.endswith('.xml'):
                parser = ET.XMLParser(encoding="utf-8")
                tree = ET.parse(os.path.join(dir_path, filename), parser=parser)
                root = tree.getroot()
                authors, _ = extract_bibliography_of_article(root)
                ref_authors.append(authors)
    except FileNotFoundError:
        print("Wrong Directory Path")
    except ET.ParseError:
        print("Parsing Error, check parsing step")
    return ref_authors


def extract_all_cited_authors_of_paper(ref_authors: list[list[list]]):
    papers_bib = []
    for pub in ref_authors:
        art = []
        for citations in pub:
            for citation in citations:
                art.append(citation)
        papers_bib.append(art)

    return papers_bib


def preprocess_author_names_article(root: ET.Element):
    author_names = []
    author_elements = root.findall('.//Author')
    for author_element in author_elements:
        try:
            name_elements = author_element.findall("./AuthorName/*")
            name = ''
            for element in name_elements:
                if element.text is not None:
                    name += element.text.strip() + ' '
            if name.strip() != '':
                # Standardize the author name by removing suffixes and extra spaces
                name = standardize_author_names(name)
            author_names.append(name)
        except Exception as e:
            print(f"Error parsing author element: {e}")

    return author_names


def create_articles_authors_info_dict(dir_path: str) -> dict[str, list[tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]]]:
    """
    Disambiguation process by grouping by affiliation
    Creating a authors dictionary mapping to authors information. Here for a same name key, various authors can be
    found in the list of values
    :param dir_path:
    :return: dict
    """
    authors_dict = {}
    for filename in os.listdir(dir_path):
        if not filename.startswith('.') and filename.endswith('.xml'):
            try:
                tree = ET.parse(os.path.join(dir_path, filename))
                root = tree.getroot()
                # print(filename)

                article_id = root.find(".//ArticleDOI")
                article_title = root.find(".//ArticleTitle")
                journal_title = root.find(".//JournalTitle")
                journal_subject = root.find(".//JournalSubject[@Type='Primary']")
                author_group_element = root.find(".//AuthorGroup")

                if author_group_element is not None:
                    author_elements = author_group_element.findall(".//Author")
                    affiliation_elements = author_group_element.findall(".//Affiliation")

                    affiliations_dict = {}
                    for affiliation_element in affiliation_elements:
                        try:
                            org_division_element = affiliation_element.find("./OrgDivision")
                            org_name_element = affiliation_element.find("./OrgName")
                            org_address_element = affiliation_element.find('./OrgAddress/*')

                            if org_division_element is None and org_name_element is None:
                                org_address = org_address_element.text.strip()
                                affiliations_dict[affiliation_element.get('ID')] = org_address
                            elif org_division_element is None:
                                org_name = org_name_element.text.strip()
                                affiliations_dict[affiliation_element.get('ID')] = org_name
                            elif org_name_element is None:
                                org_division = org_division_element.text.strip()
                                affiliations_dict[affiliation_element.get('ID')] = org_division
                            elif org_division_element is not None and org_name_element is not None:
                                org_division = org_division_element.text.strip()
                                org_name = org_name_element.text.strip()
                                affiliation = f"{org_division}, {org_name}"
                                affiliations_dict[affiliation_element.get('ID')] = affiliation
                        except Exception as e:
                            print(f"Error parsing affiliation element: {e}")

                    for author_element in author_elements:
                        try:
                            name_elements = author_element.findall("./AuthorName/*")
                            name = ''
                            for element in name_elements:
                                if element.text is not None:
                                    name += element.text.strip() + ' '
                            if name.strip() != '':
                                # name = standardize_author_names(name)
                                name = re.sub(r'\b(Jr\.|Sr\.|Ph\.D\.|MD)\b', '', name.strip()).strip()
                                affiliation_ids = author_element.get('AffiliationIDS')
                                affiliations = []
                                if affiliation_ids is not None:
                                    for affiliation_id in affiliation_ids.split(' '):
                                        if affiliation_id in affiliations_dict:
                                            affiliations.append(affiliations_dict[affiliation_id])
                                if name not in authors_dict:
                                    if len(affiliations) == 0:
                                        aff = None
                                    else:
                                        aff = affiliations[0]
                                    authors_dict[name] = [(article_id.text, article_title.text, journal_title.text,
                                                           journal_subject.text, aff)]
                                else:
                                    authors_dict[name].append((article_id.text, article_title.text, journal_title.text,
                                                               journal_subject.text, aff))
                        except Exception as e:
                            print(f"Error parsing author element: e")
            except Exception as e:
                print(f"Error parsing XML file: {filename} - {e}")
    return authors_dict


def preprocess_author_names_article(root):
    author_names = []
    author_elements = root.findall('.//Author')
    for author_element in author_elements:
        try:
            name_elements = author_element.findall("./AuthorName/*")
            name = ''
            for element in name_elements:
                if element.text is not None:
                    name += element.text.strip() + ' '
            if name.strip() != '':
                # Standardize the author name by removing suffixes and extra spaces
                name = standardize_author_names(name)
            author_names.append(name)
        except Exception as e:
            print(f"Error parsing author element: {e}")

    return author_names


def create_dataset_csv(authors_dict: dict, dir_path: str):
    df_all = pd.DataFrame()
    # loop through each author and append their articles and affiliations to the dataframe
    for author, articles in authors_dict.items():
        for article in articles:
            article_id = article[0]
            article_title = article[1]
            journal_title = article[2]
            journal_subject = article[3]
            affiliations = article[4]
            df = pd.DataFrame({
                'Author': [author],
                'ArticleID': [article_id],
                'Article Title': [article_title],
                'Journal Title': [journal_title],
                'Journal Subject': [journal_subject],
                'Affiliations': [affiliations]

            })

            df_all = pd.concat([df_all, df], ignore_index=True)

    df_all.to_csv(dir_path + 'authors.csv', index=False)


def load_data_csv(filename, delimiter):
    """
    Loading the dataset if the file exists.
    """
    try:
        if os.path.isfile(filename):
            df = pd.read_csv(filename, delimiter=delimiter)
            return df
    except FileNotFoundError as e:
        print("File not found", e)


def preprocess_dataset(input_data):
    """ The preprocessing selects the relevant data.
    :param input_data: Input data
    :return X: Feature selection for input data and returning the training data and test data.
    :rtype: .... """
    input_data = input_data.fillna("NA")

    feature_cols = ['ArticleID', 'Article Title', 'Journal Title', 'Journal Subject', 'Affiliations']
    # input_data[feature_cols] = input_data[feature_cols].apply(clean_text)
    print("Columns in the data frame", input_data.columns)
    print("Input dataframe shape", input_data.shape)

    input_data['Article Title'] = input_data['Article Title'].apply(lambda x: clean_text(x))
    input_data['Author'] = input_data['Author'].apply(lambda x: standardize_author_names(x))
    input_data['Journal Title'] = input_data['Journal Title'].apply(lambda x: clean_text(x))
    input_data['Affiliations'] = input_data['Affiliations'].apply(lambda x: clean_text(x))

    X = input_data[feature_cols]
    # X = input_data[df_all.columns[2:]]
    y = input_data['Author'].tolist()
    # Splitting the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.9,
                                                        test_size=0.1, random_state=100)

    return x_train, y_train, x_test, y_test


def transform_data():
    """
    Transforming into TF-IDF vectorization of the input data
    """
    vect = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    transformer = make_column_transformer((vect, 'ArticleID'), (vect, 'Article Title'), (vect, 'Journal Title'),
                                          (vect, 'Journal Subject'), (vect, 'Affiliations'))
    return vect, transformer


def clean_text(text):
    """
    Cleaning text such as removing punctuations, removing html tags/urls if present,
    stopwords and lowering the case
    """

    # Removing URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'', text)

    # remove html
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(html, '', text)

    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    # text = re.sub(r'([a-zA-Z0-9-/]{2,})', " ", text)

    # text = html.sub(r'', text)
    # removing punctuations
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`'
    # punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')  # Removing punctuations

    # lowercase
    text = text.lower()
    # removing stopwords
    stops = stopwords.words('english')
    # stopwords = load_stopwords('../../data/stop_words.txt')
    text = [word.lower() for word in text.split() if word.lower() not in stops]

    text = " ".join(text)

    return text


def standardize_author_names(text):
    name = re.sub(r'\b(Jr\.|Sr\.|Ph\.D\.|MD\.|[a-z]{2,}\.)\s*', '', text.strip(),
                  flags=re.IGNORECASE).strip().lower()
    return name

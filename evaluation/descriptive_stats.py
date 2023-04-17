#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import spacy
import textdescriptives as td
td.get_valid_metrics()


import textdescriptives as td
# load your favourite spacy model (remember to install it first using e.g. `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("textdescriptives/all")


def get_docs(texts, nlp, n_process=4):
    docs = nlp.pipe(texts, n_process=n_process)
    return docs

def get_descriptive_stats(docs):
    # extract all metrics
    df = td.extract_df(docs)
    return df

def clean_texts(texts):
    clean_texts = []
    for text in texts:
        text = re.sub(r"<p>", "", text)
        text = re.sub(r"</p>", r"\n\n", text)
        clean_texts.append(text.strip())
    return clean_texts

if __name__ == "__main__":

    texts = [
        """
        <p>HONOLULU, Hawaii — Many homeless people move from colder states to Hawaii where it is warm and they can sleep on the beach. David Ige is the governor of Hawaii. He says it has become a big problem. There are too many homeless people in Hawaii. Ige has declared an emergency. He says the state must help the homeless.</p><p>Ige's emergency declaration will help the state government to quickly build a homeless shelter for families.</p><p>The governor's announcement came just days after officials cleared one of the nation's largest homeless camps. As many as 300 people were living in tents. Workers took down the camp.</p><p>The workers helped 125 people find housing, including 25 families. Some people rode buses to homeless shelters. Others moved into longer-term houses, Ige said.</p><p>"They are definitely off the streets and in a better situation," he added.</p><p>## New Shelter And Special Program Will Help</p><p>Yet many people still need homes. There is especially not enough houses for families.</p><p>There are 7,260 homeless people living in Hawaii. It has the highest rate of homelessness of any state in the United States.</p><p>Scott Morishige is the state homelessness coordinator in Hawaii. The number of families without a home almost doubled in the past year, he said.</p><p>The state will spend $1.3 million to help homeless people, Morishige said. The money will pay for the new shelter. It will pay for another program called Housing First. The program provides homes and services to people who have been homeless more than once. The money will also help families pay rent.</p><p>## A Place To Stay For Now</p><p>Meanwhile, workers are putting up a new homeless shelter. This short-term shelter is located on Sand Island. The rooms have just enough space for two people.</p><p>Each room will be just as big or bigger than a tent.</p><p>The rooms were made from shipping containers. Each large rectangular container has a window and a screen door. People will be able to sit outside under a shady spot, said Russ Wozniak. He is an architect and an engineer with Group 70. His company helped designed the shelter. The units will not have air conditioning. However, a special coating on the outside will help the inside stay cool.</p><p>Nearby will be a trailer that holds five bathrooms. Each has a toilet and shower.</p><p>The shelter will be finished in December. It will house up to 87 people at a time until they find permanent homes.</p><p>
        """,
        """
        <p>HONOLULU, Hawaii — Many homeless people move to Hawaii. The islands are warm. Homeless people do not have a place to live. Many of them can sleep on the beach.</p><p>David Ige is the governor of Hawaii. He is a leader in the state. He said there are too many homeless people. David said the government must help the homeless.</p><p>There were 300 people living in a camp. They were told to leave. Workers took down the camp. The workers helped 125 people find a place to live. Some people went to homeless shelters. A shelter is a place where homeless people can stay for a short time. Others moved into houses.</p><p>They are not sleeping in the streets anymore. The homeless people are in a better place, the governor said.</p><p>## Lots Of Families Need Homes</p><p>Many families still need a house, though. A new homeless shelter must be built soon.</p><p>Scott Morishige works with Governor Ige. He helps homeless people. A million dollars will be spent to help them, Scott said. The money will buy a new shelter. It will pay for a program, too. It is called Housing First. It will help homeless people find houses and jobs. If they have jobs, they can make money to pay for a home.</p><p>## Room For Two</p><p>Workers are building a new homeless shelter. The rooms are made from big wooden shipping boxes. The rooms fit two people.</p><p>Russ Wozniak helped to plan the shelter.</p><p>Each room has a window and a screen door. It will be nice to sit outside, Russ said.</p><p>The shelter will be done in December. People will be happy to live there.</p><p>
        """,
    ]

    texts = clean_texts(texts)
    print(texts)
    docs = get_docs(texts, nlp)
    df = get_descriptive_stats(docs)
    breakpoint()
    print(df)

# access some of the values
# doc._.readability
# doc._.token_length

# text = "The world is changed. I feel it in the water. I feel it in the earth. I smell it in the air. Much that once was is lost, for none now live who remember it."
# will automatically download the relevant model (´en_core_web_lg´) and extract all metrics
# df = td.extract_metrics(text=text, lang="en", metrics=None)

# specify spaCy model and which metrics to extract
# df = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["readability", "coherence"])

# td.extract_dict(doc)
# td.extract_df(doc)
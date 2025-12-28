import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # calculating the base random jump probability for all the pages
    N = len(corpus)
    jump_rate = (1 - damping_factor) / N

    # Initialising a dictionary to store each page's jump_rate score
    probability = {}

    # the loop to iterate over the corpus in order to set a base jump rate for each page
    for p in corpus.keys():

        probability[p] = jump_rate

    # if the current page has any links to other pages distributing the damping factor probability equally amongst them
    if corpus[page]:

        # the formula for outgoing links taking the damping factor and dividing it by the number of links therefore spreading the
        # probability equally
        probability_outgoing_link = damping_factor / len(corpus[page])

        # a loop updating the probability of landing on the pages that are linked
        for linked_page in corpus[page]:

            probability[linked_page] += probability_outgoing_link

    else:

        # if there are no links then treating it as if it linking to all pages equally
        no_links_probability = damping_factor / N

        # distributing the damping factor probability across the pages equally
        for p2 in corpus.keys():

            probability[p2] += no_links_probability

    return probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # a dictionary for tracking number of visits for each page inside the loop
    visit_counts = {}

    # Iterating over the corpus to set the visit rate to a starting 0
    for key in corpus.keys():

        visit_counts[key] = 0

    # a list is used as a parameter for the rand.choice method so converting the corpus to be passed as an argument
    pages = list(corpus.keys())

    # Picking a page to start from randomly and updating visit count for that page
    start = random.choice(pages)
    visit_counts[start] += 1

    # setting the current page to the one to start from
    current_page = start

    # Looping over the pages to calculate page ranks
    for i in range(n - 1):

        # calling the transition_model() function that handles probabilities based on the outgoing links
        transition_result = transition_model(corpus, current_page, damping_factor)

        # storing the the values returned by the transition_model function as a list
        transition_probabilities = list(transition_result.values())

        # choosing the next page randomly based on the transition probabilities
        next_page = random.choices(pages, transition_probabilities, k=1)[0]

        # recording visit count
        visit_counts[next_page] += 1

        # setting the next_page as the current page for the next iteration
        current_page = next_page

    # dictionary to store the results of page rank
    pageranks = {}

    # loop to get the scores for each page based on the visit count and number of samples
    for page in visit_counts:

        rank_score = visit_counts[page] / n

        pageranks[page] = rank_score

    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # defining the empty pageranks dictionary for recording the pageranks values
    pageranks = {}

    N = len(corpus)

    # setting the baseline pagerank values for all the pages using this loop
    for page in corpus.keys():

        pageranks[page] = 1 / N

    # defining a dictionary that will hold the updated pageranks
    new_pageranks = {}

    # a while loop running until the desired level of convergence has been reached
    while True:

        # resetting the new_pageranks dictionary on each new iteration
        new_pageranks = {}

        # set has converged to True and check later if it reached the desired convergence number to decide
        # whether to iterate further
        has_converged = True

        # loop in which the pagerank scores will be calculated for each destination page
        for destination_page in corpus.keys():

            # sum of pagerank contributions from incoming links to the destination page
            sum_of_contributions = 0

            # looping over potential links to the destination page
            for potential_source_page in corpus.keys():

                links_number_psp = len(corpus[potential_source_page])

                # conditionals for checking incoming links and calculating their pagerank contributions towards the destination page
                if links_number_psp == 0:

                    sum_of_contributions += pageranks[potential_source_page] / N

                elif destination_page in corpus[potential_source_page]:

                    sum_of_contributions += pageranks[potential_source_page] / links_number_psp

            # formula for calculating the new pagerank for the current destination page
            new_pageranks[destination_page] = ((1 - damping_factor) / N) + (damping_factor * sum_of_contributions)

            # Convergence calculation and checking whether it reached the desired level
            value_dif = abs(new_pageranks[destination_page] - pageranks[destination_page])

            if value_dif >= 0.001:

                has_converged = False

        if has_converged == True:

            break

        else:

            pageranks = new_pageranks.copy()

    return pageranks


if __name__ == "__main__":
    main()

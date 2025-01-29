from neural_searcher import NeuralSearcher


def display_result(results: list):
    for i in results:
        print(i)


def main():
    neural_cosine = NeuralSearcher(collection_name="startups_cosine")

    neural_dot = NeuralSearcher(collection_name="startups_dot")

    neural_euclid = NeuralSearcher(collection_name="startups_euclid")

    query = "online platform marketing"
    query = "SaaS enterprise software security"
    query = 'Business travel booking tool'
    query = 'Recommend system for personalize'
    results = neural_cosine.search(text=query)
    print("cosine")
    display_result(results)

    results = neural_dot.search(text=query)
    print("dot")
    display_result(results)

    results = neural_euclid.search(text=query)
    print("euclid")
    display_result(results)


if __name__ == "__main__":
    main()

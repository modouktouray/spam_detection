import sys
from implementations import logistic_regression
from implementations import naive_bayes
from implementations import rnn
from implementations import svm
from implementations import baseline
# import other scripts similarly

def main(model_type):
    if model_type == "naive_bayes":
        naive_bayes.naive_bayes_main()
    elif model_type == "logistic_regression":
        logistic_regression.logistic_regression_main()
    elif model_type == "rnn":
        rnn.rnn_main()
    elif model_type == "svm":
        svm.svm_main()
    elif model_type == "baseline":
        baseline.baseline_main()
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <model_type>")
        sys.exit(1)

    model_type = sys.argv[1]
    main(model_type)

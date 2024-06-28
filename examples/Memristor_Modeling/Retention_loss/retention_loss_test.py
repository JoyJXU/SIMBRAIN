import json
import sys
sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simbrain.Fitting_Functions.retention_loss_fitting import RetentionLoss


def main():
    # Fit
    with open("../../../memristordata/my_memristor.json") as f:
        dict = json.load(f)
    file = "../../../memristordata/retention_loss.xlsx"
    exp = RetentionLoss(file)
    tau, beta = exp.fitting()
    dict.update(
        {
            "retention_loss_tau": tau,
            "retention_loss_beta": beta
        }
    )

    # Output
    df = pd.DataFrame(
        {'value': [tau, beta]},
        index=['retention_loss_tau', 'retention_loss_beta']
    )
    print(df)
    with open("fitting_record.json", "w") as f:
        json.dump(dict, f, indent=2)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(exp.time * exp.delta_t, exp.conductance, c='r')
    plt.plot(exp.time * exp.delta_t, exp.retention_loss(exp.time, tau, beta), c='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Conductance (S)")
    plt.title("Retention Loss")

    plt.tight_layout()
    plt.savefig("Retention Loss.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

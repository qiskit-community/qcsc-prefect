# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com), on miyabi-observability repo
# Modified: Yuto Morohoshi (mrhsyut@ibm.com), to migrate from miyabi-observability to qii-miyabi-kawasaki repo


from src import riken_sqd_de


if __name__ == "__main__":
    riken_sqd_de.serve(
        description="SQD with LUCJ parameter optimization with differential evoluation.",
    )

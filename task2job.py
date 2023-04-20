def task2job(task):
    """
    Converts task_id from slurm and outputs desired experiment folder.
    When specifying folders from scratch, may be usefull to
    """

    doPrefix = False
    prefix = '372564'
    job = task // 1000
    job_ids = [
        # 0
        37256432,
        # 1
        37256433,
        # 2
        37256452,
        # 3
        37256435,
        # 4
        37256434,

        # 5
        37256442,
        # 6
        37256456,
        # 7
        37256443,
        # 8
        37256458,
        # 9
        37256445,

        # 10
        37256436,
        # 11
        37256444,
        # 12
        37256462,
        # 13
        37256439,
        # 14
        37256464,

        # 15
        37256438,
        # 16
        37256466,
        # 17
        37256441,
        # 18
        37256458,
        # 19
        37256459,

        # 20
        37256600,
        # 21
        37256601,
        # 22
        37256440,
        # 23
        37256446,
        # 24
        37256604,

        # 25
        37256437,
        # 26
        37256646,
        # 27
        37256647,
        # 28
        37256648,
        # 29
        37256649,

        # 30
        37256540,
        # 31
        37256541,
        # 32
        37256542,
        # 33
        37256447,
        # 34
        37256544,

        # 35
        37256545,
    ]

    experiment_id = "37256396"
    return prefix*doPrefix + str(job_ids[job]), experiment_id


# print(task2job(13000))
# print(task2job(1300))

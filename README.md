# fluent-cv
A fluent style interface intended for working with images using opencv. Many times you may want to apply a couple filters, tweak the settings and compare results. Keeping track of changes, adding preview, timing or saving can take a few seconds and lines of code that distracts from the process.

Fluent_cv provides a simple way to apply functions, preview and/or save the result and keep track of which parameters were used to generate which result, e.g.:

    import cv2
    from fluent_cv import ImageProcessor

    processor = ImageProcessor().load_image("data/test.jpg")

    (processor
            .apply_filter_function(cv2.normalize, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            .apply_filter_function(cv2.medianBlur, 5)
            .preview()
            .apply_filter_function(cv2.bilateralFilter, 11, 75, 75)
            .optional_save()  # save y/n, add tags if desired
    )

Save results with steps and settings for quick comparison, e.g.:

    test_edited_revision_1.json:

    {
        "original_filepath_loaded": "data/test.jpg",
        "timestamp": "2024-12-15T11:30:21.025135",
        "processing_steps": [
            {
                "filter_name": "loaded image: data/test.jpg"
            },
            {
                "filter_time": 0.000023,
                "filter_name": "normalize",
                "parameters": {
                    "args": [
                        "None"
                    ],
                    "kwargs": {
                        "alpha": "0",
                        "beta": "255",
                        "norm_type": "32"
                    }
                }
            },
            {
                "filter_time": 0.001002,
                "filter_name": "medianBlur",
                "parameters": {
                    "args": [
                        "5"
                    ],
                    "kwargs": {}
                }
            },
            {
                "filter_time": 0.002997,
                "filter_name": "bilateralFilter",
                "parameters": {
                    "args": [
                        "11",
                        "75",
                        "75"
                    ],
                    "kwargs": {}
                }
            }
        ]
    }
(define (problem Read_book)
    (:domain virtualhome)
    (:objects
    character - character
    bathroom novel home_office - object
)
    (:init
    (movable novel)
    (cuttable novel)
    (obj_inside novel home_office)
    (inside character bathroom)
    (sitting character)
    (has_paper novel)
    (grabbable novel)
    (readable novel)
    (can_open novel)
)
    (:goal
    (and
        (holds_rh character novel)
    )
)
    )
    
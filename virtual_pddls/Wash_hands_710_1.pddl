(define (problem Wash_hands)
    (:domain virtualhome)
    (:objects
    character - character
    home_office hands_both water bathroom sink bathroom_counter soap - object
)
    (:init
    (surfaces bathroom_counter)
    (containers sink)
    (recipient sink)
    (drinkable water)
    (pourable water)
    (cream soap)
    (grabbable soap)
    (movable soap)
    (body_part hands_both)
    (inside character home_office)
    (next_to sink soap)
    (next_to soap sink)
)
    (:goal
    (and
    )
)
    )
    
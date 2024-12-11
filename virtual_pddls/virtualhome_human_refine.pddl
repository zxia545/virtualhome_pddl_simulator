(define (domain virtualhome)
    (:requirements :typing)
    (:types object character)
    (:predicates
        (closed ?obj - object)
        (open ?obj - object)
        (on ?obj - object)
        (off ?obj - object)
        (plugged_in ?obj - object)
        (plugged_out ?obj - object)
        (sitting ?char - character)
        (lying ?char - character)
        (clean ?obj - object)
        (dirty ?obj - object)
        (obj_ontop ?obj1 - object ?obj2 - object)
        (ontop ?char - character ?obj - object)
        (on_char ?obj - object ?char - character)
        (inside_room ?obj - object ?room - object)
        (obj_inside ?obj1 - object ?obj2 - object)
        (inside ?char - character ?obj - object)
        (obj_next_to ?obj1 - object ?obj2 - object)
        (next_to ?char - character ?obj - object)
        (between ?obj1 - object ?obj2 - object ?obj3 - object)
        (facing ?char - character ?obj - object)
        (holds_rh ?char - character ?obj - object)
        (holds_lh ?char - character ?obj - object)
        (grabbable ?obj - object)
        (cuttable ?obj - object)
        (can_open ?obj - object)
        (readable ?obj - object)
        (has_paper ?obj - object)
        (movable ?obj - object)
        (pourable ?obj - object)
        (cream ?obj - object)
        (has_switch ?obj - object)
        (lookable ?obj - object)
        (has_plug ?obj - object)
        (drinkable ?obj - object)
        (body_part ?obj - object)
        (recipient ?obj - object)
        (containers ?obj - object)
        (cover_object ?obj - object)
        (surfaces ?obj - object)
        (sittable ?obj - object)
        (lieable ?obj - object)
        (person ?obj - object)
        (hangable ?obj - object)
        (clothes ?obj - object)
        (eatable ?obj - object)
        (has_free_hand ?char - character)

    )

    ; (:actions walk_towards walk_into sit standup grab open_this close put_on put_on_character put_inside switch_on switch_off turn_to wipe drop lie pour watch move wash plug_in plug_out )
    (:action walk_towards
        :parameters (?char - character ?obj - object)
        :precondition (and
            (not (sitting ?char))
            (not (lying ?char))
        )
        :effect (and
            ; Remove all existing (next_to ?char ?obj2)
            (forall (?obj2 - object)
                (not (next_to ?char ?obj2))
            )
            ; Add (next_to ?char ?obj)
            (next_to ?char ?obj)
        )
    )


    (:action walk_into
        :parameters (?char - character ?room - object)
        :precondition (and
            (not (sitting ?char))
            (not (lying ?char))
        )
        :effect (and
            ; Remove all existing (inside ?char ?room2)
            (forall (?room2 - object)
                (not (inside ?char ?room2))
            )
            ; Add (inside ?char ?room)
            (inside ?char ?room)
            ; Remove all (next_to ?char ?obj2)
            (forall (?obj2 - object)
                (not (next_to ?char ?obj2))
            )
        )
    )


    
    
    (:action sit
        :parameters (?char - character ?obj - object)
        :precondition (and
            (next_to ?char ?obj)
            (sittable ?obj)
            (not (sitting ?char))
        )
        :effect (and
            (sitting ?char)
            (ontop ?char ?obj)
        ) 
    )
    
    (:action standup
        :parameters (?char - character)
        :precondition (or 
            (sitting ?char)
            (lying ?char)
        )
        :effect (and 
            (not (sitting ?char))
            (not (lying ?char))
        )
    )

    (:action grab
        :parameters (?char - character ?obj - object)
        :precondition (and
            (grabbable ?obj)
            (next_to ?char ?obj)
            (not (exists (?obj2 - object) (and (obj_inside ?obj ?obj2) (closed ?obj2))))
            (not (and (exists (?obj3 - object) (holds_lh ?char ?obj3)) (exists (?obj4 - object) (holds_rh ?char ?obj4))))
        )
        :effect (and
            (when (exists (?obj3 - object) (holds_lh ?char ?obj3)) (holds_rh ?char ?obj))
            (when (exists (?obj4 - object) (holds_rh ?char ?obj4)) (holds_lh ?char ?obj))
            (when 
                (not (and (exists (?obj3 - object) (holds_lh ?char ?obj3)) (exists (?obj4 - object) (holds_rh ?char ?obj4))))
                (holds_rh ?char ?obj)
            )
        ) 
    )




    (:action open_this
        :parameters (?char - character ?obj - object)
        :precondition (and
            (can_open ?obj)
            (closed ?obj)
            (next_to ?char ?obj)
            (not (on ?obj))
        )  
        :effect (and
            (open ?obj)
            (not (closed ?obj))
        ) 
    )
    
    (:action close
        :parameters (?char - character ?obj - object)
        :precondition (and
            (can_open ?obj)
            (open ?obj)
            (next_to ?char ?obj)
        )
        :effect (and
            (closed ?obj)
            (not (on ?obj))
        ) 
    )
    
    (:action put_on
        :parameters (?char - character ?obj1 - object ?obj2 - object)
        :precondition (or
            (and
                (next_to ?char ?obj2)
                (holds_lh ?char ?obj1)
            )
            (and
                (next_to ?char ?obj2)
                (holds_rh ?char ?obj1)
            )
        )
        :effect (and
            (obj_next_to ?obj1 ?obj2)
            (obj_ontop ?obj1 ?obj2)
            (not (holds_lh ?char ?obj1))
            (not (holds_rh ?char ?obj1))
        )
    )
    
    (:action put_on_character
        :parameters (?char - character ?obj - object)
        :precondition (or
            (holds_lh ?char ?obj)
            (holds_rh ?char ?obj)
        )
        :effect (and
            (on_char ?obj ?char)
            (not (holds_lh ?char ?obj))
            (not (holds_rh ?char ?obj))
        )
    )
    
    (:action put_inside
        :parameters (?char - character ?obj1 - object ?obj2 - object)
        :precondition (or
            (and
                (next_to ?char ?obj2)
                (holds_lh ?char ?obj1)
                (not (can_open ?obj2))
            )
            (and
                (next_to ?char ?obj2)
                (holds_lh ?char ?obj1)
                (open ?obj2)
            )
            (and
                (next_to ?char ?obj2)
                (holds_rh ?char ?obj1)
                (not (can_open ?obj2))
            )
            (and
                (next_to ?char ?obj2)
                (holds_rh ?char ?obj1)
                (open ?obj2)
            )
        )
        :effect (and
            (obj_inside ?obj1 ?obj2)
            (not (holds_lh ?char ?obj1))
            (not (holds_rh ?char ?obj1))
        ) 
    )
    
    (:action switch_on
        :parameters (?char - character ?obj - object)
        :precondition (and
            (has_switch ?obj)
            (off ?obj)
            (plugged_in ?obj)
            (next_to ?char ?obj)      
        )  
        :effect (and
            (on ?obj)
            (not (off ?obj))
        )
    )
    
    (:action switch_off
        :parameters (?char - character ?obj - object)
        :precondition (and
            (has_switch ?obj)
            (on ?obj)
            (next_to ?char ?obj)                  
        )  
        :effect (and
            (off ?obj)
            (not (on ?obj))
        )
    )
    
    (:action wipe
        :parameters (?char - character ?obj1 - object ?obj2 - object)
        :precondition (or
            (and 
                (next_to ?char ?obj1) 
                (holds_lh ?char ?obj2)
            )
            (and 
                (next_to ?char ?obj1) 
                (holds_rh ?char ?obj2)
            )
        )
        :effect (and 
            (clean ?obj1)
            (not (dirty ?obj1))
        )
    )
    
    (:action drop
        :parameters (?char - character ?obj - object ?room - object)
        :precondition (or
            (and 
                (holds_lh ?char ?obj)
                (obj_inside ?obj ?room)
            )
            (and 
                (holds_rh ?char ?obj)
                (obj_inside ?obj ?room)
            )
        )               
        :effect (and
            (not (holds_lh ?char ?obj))
            (not (holds_rh ?char ?obj))
        ) 
    )
    
    (:action lie 
        :parameters (?char - character ?obj - object)
        :precondition (and 
            (lieable ?obj) 
            (next_to ?char ?obj)
            (not (lying ?char))
        )
        :effect (and
            (lying ?char)
            (ontop ?char ?obj)
            (not (sitting ?char))
        )
    )
    
    (:action pour 
        :parameters (?char - character ?obj1 - object ?obj2 - object)
        :precondition (or
            (and 
                (pourable ?obj1) 
                (holds_lh ?char ?obj1)
                (recipient ?obj2)
                (next_to ?char ?obj2)
            )
            (and 
                (pourable ?obj1) 
                (holds_rh ?char ?obj1)
                (recipient ?obj2)
                (next_to ?char ?obj2)
            )
            (and 
                (drinkable ?obj1) 
                (holds_lh ?char ?obj1)
                (recipient ?obj2)
                (next_to ?char ?obj2)
            )
            (and 
                (drinkable ?obj1) 
                (holds_rh ?char ?obj1)
                (recipient ?obj2)
                (next_to ?char ?obj2)
            )
        )
        :effect (obj_inside ?obj1 ?obj2)
    )
    
    (:action wash 
        :parameters (?char - character ?obj - object)
        :precondition (and 
            (next_to ?char ?obj)
        )
        :effect (and
            (clean ?obj)
            (not (dirty ?obj))
        )
    )
    
    (:action plug_in 
        :parameters (?char - character ?obj - object)
        :precondition (or
            (and 
                (next_to ?char ?obj)
                (has_plug ?obj)
                (plugged_out ?obj)
            )
            (and 
                (next_to ?char ?obj)
                (has_switch ?obj)
                (plugged_out ?obj)
            )
        )
        :effect (and
            (plugged_in ?obj)
            (not (plugged_out ?obj))
        )
    )
    
    (:action plug_out 
        :parameters (?char - character ?obj - object)
        :precondition (and 
            (next_to ?char ?obj)
            (has_plug ?obj)
            (plugged_in ?obj)
            (not (on ?obj))
        )
        :effect (and
            (plugged_out ?obj)
            (not (plugged_in ?obj))
        )
    )
)
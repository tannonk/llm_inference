# Input format

The expected input format for any given dataset is a JSON-lines file (jsonl).

Each line should contain a dictionary-like object with keys `complex` and `simple`.

`simple` should be defined as a `list` to be compatible with datasets containing multiple reference texts.

Here's an example from ASSET:

```json
{
    "complex": "One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.", 
    "simple": [
        "On one side of the conflicts are the Sudanese military and the Janjaweed, a Sudanese militia group.  They are mostly recruited from the Afro-Arab Abbala tribes.", 
        "One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed. The Janjaweed are a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.", 
        "One side of the armed conflicts is mainly the Sudanese military and the Janjaweed militia group.", 
        "One side of the war is made up of the Sudanese military and the Janjaweed, a Sudanese militia group from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.", 
        "One side of the war is mainly made up of the Sudanese military and the Janjaweed. The Janjaweed is a Sudanese militia group who come mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.", 
        "One side of the fighters is made up of the Sudanese military and the Janjaweed.  The Janjaweed is a Sudanese militia group.  It is recruited mostly from the Afro-Arab Abbala tribes located in the northern Rizeigat region of Sudan.", 
        "One side of the armed conflict includes the Sudanese military and the Janjaweed, a Sudanese militia group made up of Afro-Arab Abbala tribesmen.", 
        "One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed.  The Janjaweed are a Sudanese militia group, with recruits from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.", 
        "One side of the war is mainly the Sudanese military and a Sudanese milita group called the Janjaweed. The Janjaweed is mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.", 
        "The Sudanese military and Janjaweed work together. The Janjaweed is a militia group. The members were recruited from Afro-Arab Abbala tribes. These tribes are from the northern region of Sudan."
    ]
}
```

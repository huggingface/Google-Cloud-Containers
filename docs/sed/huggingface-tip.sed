/^> \[!NOTE\]/{
s//<Tip>\
/
p
:note_loop
n
/^$/!{
    s/^> //
    p
    b note_loop
}
s/^$/\
<\/Tip>\
/
p
b
}
/^> \[!WARNING\]/{
s//<Tip warning>\
/
p
:warning_loop
n
/^$/!{
    s/^> //
    p
    b warning_loop
}
s/^$/\
<\/Tip>\
/
p
b
}
p

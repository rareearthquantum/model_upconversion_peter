function x = linblad(m)
    x = 2*spre(m)*spost(m')-spre(m'*m)-spost(m'*m);
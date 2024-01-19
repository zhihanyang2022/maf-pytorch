<TeXmacs|2.1.1>

<style|generic>

<\body>
  Given that

  <\equation*>
    P<around*|(|A<text| win>\<mid\>B<text| scores first>|)>=0.3.
  </equation*>

  So

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|P<around*|(|A<text| did not
    score first>\<mid\>A<text| win>|)>>>|<row|<cell|>|<cell|=>|<cell|1-P<around*|(|A<text|
    scored first>\<mid\>A<text| win>|)>-<wide*|P<around*|(|<text|Nobody
    scored first>\<mid\>A<text| win>|)>|\<wide-underbrace\>><rsub|0>>>|<row|<cell|>|<cell|=>|<cell|1-<frac|P<around*|(|A<text|
    scored first> \<cap\>A<text| win>|)>|P<around*|(|A<text|
    win>|)>>>>|<row|<cell|>|<cell|=>|<cell|1-<frac|0.48|0.48+0.3\<times\>0.35>>>|<row|<cell|>|<cell|\<approx\>>|<cell|0.179>>>>
  </eqnarray*>

  MOG likelihood

  <\eqnarray*>
    <tformat|<table|<row|log p<around*|(|x|)>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>\<pi\><rsub|c>
    <with|font|cal|N><around*|(|x\<mid\>\<mu\><rsub|c>,\<sigma\><rsub|c>|)>|}>>>|<row|<cell|>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c>+log <with|font|cal|N><around*|(|x\<mid\>\<mu\><rsub|c>,\<sigma\><rsub|c>|)>|)>|}>>>|<row|<cell|>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c>+log <around*|{|<frac|1|\<sigma\><rsub|c><sqrt|2\<pi\>>>e<rsup|-<frac|1|2><around*|(|<frac|x-\<mu\><rsub|c>|\<sigma\><rsub|c>>|)><rsup|2>>|}>|)>|}>>>|<row|<cell|>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c>-log \<sigma\><rsub|c>-<frac|1|2> log
    2\<pi\>-<frac|1|2><around*|(|<frac|x-\<mu\><rsub|c>|\<sigma\><rsub|c>>|)><rsup|2>|)>|}>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|>|<cell|<rsub|>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|=>|<cell|log<big|sum><rsub|c=1><rsup|C>\<pi\><rsub|c><around*|(|None|)><big|prod><rsub|d=1><rsup|D><with|font|cal|N><around*|(|x<rsub|d><rsup|<around*|(|i|)>>\<mid\><with|font-series|bold|\<mu\>><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>,<with|font-series|bold|\<sigma\>><rsup|2><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>|)>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|=>|<cell|log<big|sum><rsub|c=1><rsup|C>\<pi\><rsub|c><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)><big|prod><rsub|d=1><rsup|D><with|font|cal|N><around*|(|x<rsub|d><rsup|<around*|(|i|)>>\<mid\><with|font-series|bold|\<mu\>><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>,<with|font-series|bold|\<sigma\>><rsup|2><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|log<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>+<big|sum><rsub|d=1><rsup|D>log
    <with|font|cal|N><around*|(|x<rsub|d><rsup|<around*|(|i|)>>\<mid\><with|font-series|bold|\<mu\>><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>,<with|font-series|bold|\<sigma\>><rsup|2><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|log<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>+<big|sum><rsub|d=1><rsup|D><around*|[|-log
    \<sigma\><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>-<frac|1|2>
    log <around*|(|2\<pi\>|)>-<frac|1|2><around*|(|<frac|x<rsub|d>-\<mu\><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>|\<sigma\><rsub|c,d><around*|(|<with|font-series|bold|x><rsup|<around*|(|i|)>>|)>>|)><rsup|2>|]>|)>>>>>
  </eqnarray*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>
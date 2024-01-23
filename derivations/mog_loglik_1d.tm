<TeXmacs|2.1.1>

<style|generic>

<\body>
  One-dimensional mixture of Gaussians density function:

  <\equation*>
    p<around*|(|x|)>=<big|sum><rsub|c=1><rsup|C>\<pi\><rsub|c>
    <with|font|cal|N><around*|(|x\<mid\>\<mu\><rsub|c>,\<sigma\><rsub|c>|)>
  </equation*>

  Log-likelihood function:

  <\eqnarray*>
    <tformat|<table|<row|log p<around*|(|x|)>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>\<pi\><rsub|c>
    <with|font|cal|N><around*|(|x\<mid\>\<mu\><rsub|c>,\<sigma\><rsub|c>|)>|}>>>|<row|<cell|>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c>+log <with|font|cal|N><around*|(|x\<mid\>\<mu\><rsub|c>,\<sigma\><rsub|c>|)>|)>|}>>>|<row|<cell|>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c>+log <around*|{|<frac|1|\<sigma\><rsub|c><sqrt|2\<pi\>>>e<rsup|-<frac|1|2><frac|<around*|(|x-\<mu\><rsub|c>|)><rsup|2>|\<sigma\><rsub|c><rsup|2>>>|}>|)>|}>>>|<row|<cell|>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c>+log <frac|1|\<sigma\><rsub|c>>+log
    <frac|1|<sqrt|2\<pi\>>>-<frac|1|2><frac|<around*|(|x-\<mu\><rsub|c>|)><rsup|2>|\<sigma\><rsub|c><rsup|2>>|)>|}>>>|<row|<cell|>|<cell|=>|<cell|log<around*|{|<big|sum><rsub|c=1><rsup|C>exp<around*|(|log
    \<pi\><rsub|c>+0.5 log <around*|(|<frac|1|\<sigma\><rsub|c><rsup|2>><rsup|>|)>-0.5log
    2\<pi\>-<frac|1|2><around*|(|x-\<mu\><rsub|c>|)><rsup|2><around*|(|<frac|1|\<sigma\><rsub|c><rsup|2>>|)>|)>|}>>>>>
  </eqnarray*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>
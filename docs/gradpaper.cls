%%% ufpethesis.cls
%%% Based on http://www.cin.ufpe.br/~paguso/ufpethesis/

\ProvidesClass{gradpaper}[2014/11/29]


%% OPTIONS

\DeclareOption{en}{
  \PassOptionsToPackage{brazil,english}{babel}
}

% Default options
\ExecuteOptions{en}
\ProcessOptions

\LoadClass[12pt,a4paper]{report}


%% PACKAGES

\RequirePackage{amsmath,amsfonts,amssymb,amsthm}
\RequirePackage[utf8]{inputenc}
\RequirePackage[brazil,english]{babel}
\RequirePackage{calc}
\RequirePackage{ifthen}
\RequirePackage{textcase}
\RequirePackage{textcomp}
\RequirePackage{url}
\RequirePackage{xspace}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}
\RequirePackage[scaled=0.92]{helvet}
\RequirePackage{courier}
\RequirePackage[pdftex]{graphicx}
\RequirePackage{epstopdf}
\RequirePackage[a4paper,margin=2cm]{geometry}
\RequirePackage{afterpage}

% Definitions
\graphicspath{{images/}}


%% GENERAL PURPOSE MACROS

\let\origcleardoublepage=\cleardoublepage
\def\cleardoublepage{
  \newpage{\pagestyle{empty}\origcleardoublepage}
}

% For use with the pseudocode package
\def\@lopcchapterspace{
  \relax
}


%% LABELS

\gdef\@presentationtext{
Paper presented to the \@degree\space Program at \@institute\space of \@university
\space as partial requirement for the degree in \@majorfield.
}
\gdef\@signaturestext{
This study, entitled {\regulartextit\@title}, is the result of the academic work developed by
student \@author, under the guidance of prof. \@adviser. The following people
recognize this \@papertype\space and it's results.
}
\gdef\abstrname{Abstract}
\gdef\keywordsname{Keywords}
\gdef\@notdefined{UNDEFINED}
\gdef\signaturesname{Signatures}
\gdef\acknowledgementsname{Acknowledgements}
\gdef\@axiomname{Axiom}
\gdef\@conjecturename{Conjecture}
\gdef\@defname{Definition}
\gdef\@lemmaname{Lemma}
\gdef\@theoname{Theorem}
\gdef\@propname{Proposition}
\gdef\@corname{Corollary}
\gdef\@proofname{Proof}
\gdef\@figurename{Figure}
\gdef\@tablename{Table}
\gdef\@equationame{equation}
\gdef\@chaptername{Chapter}
\gdef\@sectionname{Section}
\gdef\@appendixname{Appendix}
\gdef\@pagename{page}
\gdef\@authorname{Author}
\gdef\@advisername{Adviser}
\gdef\@revisername{Reviser}


%% IDENTIFICATION

% School identification
\def\university#1{
  \gdef\@university{#1}
}
\def\@university{\@notdefined}
\def\universitylogo{
  \includegraphics{ufpelogo}
}
\def\@universitylogo{\@notdefined}
\def\institute#1{
  \gdef\@institute{#1}
}
\def\@institute{\@notdefined}
\def\institutelogo{
  \includegraphics{cinlogo}
}
\def\@institutelogo{\@notdefined}
\def\address#1{
  \gdef\@address{#1}}
\def\@address{\@notdefined}

% Degree identification
\def\degree#1{
  \gdef\@degree{#1}
}
\def\@degree{\@notdefined}
\def\majorfield#1{
  \gdef\@majorfield{#1}}
\def\@majorfield{\@notdefined}
\def\@program{\@majorfield\space \@degree\space Program}

% Author identification
\def\author#1{
  \gdef\@author{#1}
}
\def\@author{\@notdefined}
\def\adviser#1{
  \gdef\@adviser{#1}
}
\def\@adviser{\@notdefined}
\def\advisertitle#1{
  \gdef\@advisertitle{#1}
}
\def\@advisertitle{\@notdefined}
\def\reviser#1{
  \gdef\@reviser{#1}
}
\def\@reviser{\@notdefined}
\def\revisertitle#1{
  \gdef\@revisertitle{#1}
}
\def\@revisertitle{\@notdefined}

% Work identification
\def\title#1{
  \gdef\@title{#1}
}
\def\@title{\@notdefined}
\def\papertype#1{
  \gdef\@papertype{#1}
}
\def\@papertype{Major Paper}


%% STYLE

% Fonts
\newcommand\regulartext{\normalfont\normalsize}
\newcommand\regulartextit{\it\normalsize}

% Frontpage
\def\frontpage{
  \clearpage
  \thispagestyle{empty}
  \begin{center}
    \input\universitylogo
    \\
    \input\institutelogo
    \large
    \\
    \@university
    \\
    \@institute
    \vfill
    \begin{minipage}{120mm}
      \begin{center}
        {\Large\bfseries\@title}
        \vskip\baselineskip
        \@papertype
      \end{center}
    \end{minipage}
    \\
    \vfill
    \begin{minipage}{100mm}
      \regulartext
      \begin{center}
        \begin{flushleft}
          \@authorname:\space\@author
          \\
          \@advisername:\space\@adviser,\space\@advisertitle
        \end{flushleft}
      \end{center}
    \end{minipage}
    \vfill
    \regulartext
    \@address
    \\
    \@date
  \end{center}
}

% Presentationpage
\def\presentationpage{
  \clearpage
  \thispagestyle{empty}
  \begin{center}
    \begin{minipage}{120mm}
      \begin{center}
        {\Large\bfseries\@title}
      \end{center}
    \end{minipage}
    \vfill
    \begin{flushright}
      \begin{minipage}{80mm}
        \regulartextit
        \@presentationtext
        \vskip\baselineskip
        \regulartext
      \end{minipage}
    \end{flushright}
    \vfill
    \vfill
    \begin{minipage}{100mm}
      \begin{flushleft}
        \@authorname:\space\@author
        \\
        \@advisername:\space\@adviser,\space\@advisertitle
        \\
        \@revisername:\space\@reviser,\space\@revisertitle
      \end{flushleft}
    \end{minipage}
    \vskip4\baselineskip
    \@address
    \\
    \@date
  \end{center}
}

% Signatures
\def\signatures{
  \clearpage
  \chapter*{\signaturesname}
  \thispagestyle{empty}
  \@signaturestext
  \begin{center}
    \vfill
    \rule{0.7\textwidth}{.4pt}
    \\
    \@author
    \vfill
    \rule{0.7\textwidth}{.4pt}
    \\
    \@adviser
    \vfill
    \rule{0.7\textwidth}{.4pt}
    \\
    \@reviser
    \vfill
    \regulartext
    \@address
    \\
    \@date
  \end{center}
}

% Acknowledgements
% TODO aumentar o espaço entre as linhas
\def\acknowledgements#1{
  \clearpage
  \chapter*{\acknowledgementsname}
  \thispagestyle{empty}
  \begin{minipage}{110mm}
    #1
  \end{minipage}
}

% Epigraph and quotations
\def\epigraph{
  \clearpage
  \thispagestyle{empty}
  ~\\\vfill
  \begin{quotation}
}
\def\endepigraph{
  \end{quotation}
  \vfill
}

\def\quotation{
  \@ifnextchar [
    {\begin{pquotation}}
    {\begin{pquotation}[\@empty]}
}
\def\endquotation{
  \end{pquotation}\@afterindentfalse\@afterheading
}

\def\pquotation[#1]#2{
  \def\@qauthor{#2}
  \def\@qnote{#1}
  \begin{flushright}
    \begin{minipage}{0.8\textwidth}
      \begin{flushright}\regulartextit
}
\def\endpquotation{
        \vskip.2\baselineskip
        \regulartext---\MakeTextUppercase{\@qauthor}
        \if\@qnote\@empty
          \relax
        \else
          \space(\@qnote)
        \fi
      \end{flushright}
    \end{minipage}
  \end{flushright}
  \normalfont\vskip2\baselineskip
}

% Abstract and keywords
\def\abstract{
  \clearpage
  \gdef\@keywordsname{\keywordsname}
  \chapter*{\abstrname}
  \thispagestyle{empty}
}

\def\@keywordsname{\@defaultkeywordsname}
\def\keywords{
  \par\vskip\baselineskip\noindent{\bf\@keywordsname: }
}
\def\endkeywords{}

% Table of contents
\def\@plaintocline#1#2#3#4#5{
  \ifnum #1>\c@tocdepth \else
    \vskip \z@ \@plus.2\p@
    {\leftskip #2\relax \rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
     {#4}\nobreak
     \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{}\mkern \@dotsep
        mu$}\hfill
     \nobreak
     \hb@xt@\@pnumwidth{\hfil\normalfont \normalcolor #5}
     \par}
  \fi
}

\renewcommand\tableofcontents{
  \clearpage
  \chapter*{\contentsname}
  \thispagestyle{empty}
  \@starttoc{toc}
}


% \def\l@part#1#2{
%   \ifnum \c@tocdepth >-2\relax
%     \addpenalty{-\@highpenalty}
%     \addvspace{2.25em \@plus\p@}
%     \setlength\@tempdima{3em}
%     \begingroup
%       \parindent \z@ \rightskip \@pnumwidth
%       \parfillskip -\@pnumwidth
%       {\leavevmode
%        \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
%        \nobreak
%          \global\@nobreaktrue
%          \everypar{\global\@nobreakfalse\everypar{}}
%     \endgroup
%   \fi}

\def\l@chapter#1#2{
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}
    \vskip .75em \@plus\p@
    \setlength\@tempdima{1.5em}
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
    %\vskip .75em \@plus\p@
  \fi}

%\def\l@section{\@plaintocline{1}{1.5em}{2.3em}}
%\def\l@subsection{\@plaintocline{2}{3.8em}{3.2em}}
%\def\l@subsubsection{\@plaintocline{3}{7.0em}{4.1em}}
%\def\l@paragraph{\@plaintocline{4}{10em}{5em}}
%\def\l@subparagraph{\@plaintocline{5}{12em}{6em}}
%
%\def\l@figure{\@plaintocline{1}{1.5em}{2.3em}}
%\def\l@table{\@plaintocline{1}{1.5em}{2.3em}}
\let\@dottedtocline\@plaintocline

\setcounter{tocdepth}{3}


\renewcommand\listoffigures{
   \clearpage
   \chapter*{\listfigurename}
   \thispagestyle{empty}
   \@starttoc{lof}}

\renewcommand\listoftables{
   \clearpage
   \chapter*{\listtablename}
   \thispagestyle{empty}
   \@starttoc{lot}}


% Sectioning
\setcounter{secnumdepth}{4}

\def\part{
	\if@openright\cleardoublepage\else\clearpage\fi
  \thispagestyle{empty}
  \secdef\@part\@spart
}
\def\@part[#1]#2{
  \ifnum \c@secnumdepth >-2\relax
    \refstepcounter{part}
    \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}
  \else
    \addcontentsline{toc}{part}{#1}
  \fi
  \markboth{}{}
  {\centering
   \interlinepenalty \@M
   \normalfont
   \null\vfil
   \ifnum \c@secnumdepth >-2\relax
     {\sc\Large\partname\nobreakspace\thepart}
     \par
     \vskip 20\p@
   \fi
   {\huge\bfseries #2\par}}
   \vfil
}
\def\@spart#1{
  {\centering
   \interlinepenalty \@M
   \normalfont
   \null\vfil
   {\huge\bfseries #1\par}}
   \vfil
}

\def\chapter{
  \if@openright\cleardoublepage\else\clearpage\fi
  \thispagestyle{plain}
  \global\@topnum\z@
  \@afterindentfalse
  \secdef\@chapter\@schapter
}

\def\@chapter[#1]#2{
  \refstepcounter{chapter}
  \addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}#1}
  \chaptermark{#1}
  \addtocontents{lof}{\protect\addvspace{10\p@}}
  \addtocontents{lot}{\protect\addvspace{10\p@}}
  \@lopcchapterspace
  \@makechapterhead{#2}
  \@afterheading
}

\def\@makechapterhead#1{
  \global\topskip 1.5pc
  \relax
  \begingroup
  \LARGE\bfseries\centering
    \ifnum\c@secnumdepth>\m@ne
      \leavevmode \hskip-\leftskip
      \rlap{\vbox to\z@{\vss
          \centerline{\large\sc{\chaptername}\enspace\thechapter}
          \vskip2.5pc}}\hskip\leftskip\fi
     #1\par \endgroup
  \skip@10pc \advance\skip@-\normalbaselineskip
  \vskip\skip@
}

\def\@schapter#1{
  \schaptermark{#1}
  \@makeschapterhead{#1}
  \@afterheading
}

\def\@makeschapterhead#1{
  \global\topskip 1.5pc\relax
  \begingroup
  \LARGE\bfseries\centering
  #1\par \endgroup
  \skip@10pc \advance\skip@-\normalbaselineskip
  \vskip\skip@
}

\def\appendix{
  \setcounter{chapter}{0}
  \renewcommand{\thechapter}{\Alph{chapter}}
  \renewcommand{\chaptername}{\appendixname}
}

\def\@startsection#1#2#3#4#5#6{
  \if@noskipsec \leavevmode \fi
  \par \@tempskipa #4\relax
  \@afterindentfalse
  \ifdim \@tempskipa <\z@ \@tempskipa -\@tempskipa \@afterindentfalse\fi
  \if@nobreak
    \everypar{}
  \else
    \addpenalty\@secpenalty\addvspace\@tempskipa
  \fi
  \@ifstar{\@dblarg{\@sect{#1}{\@m}{#3}{#4}{#5}{#6}}}
          {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}
}

\def\section{
  \@startsection{section}{1}{0mm}{1.5\baselineskip}
    {\baselineskip}{\centering\large\bfseries}
}

\def\subsection{
  \@startsection{subsection}{2}{0mm}{1.2\baselineskip}
    {.6\baselineskip}{\centering\bfseries}
}

\def\subsubsection{
  \@startsection{subsubsection}{3}{0mm}{\baselineskip}
   {.6\baselineskip}{\centering\normalfont}
}

\def\paragraph{
  \@startsection{paragraph}{4}{0mm}{\baselineskip}
   {-1em}{\itshape}
}

\def\colophon{
  \if@openright\cleardoublepage\else\clearpage\fi
  \thispagestyle{empty}
  \null\vfill
  \small\noindent
  \@colophontext
}


% Headers & footers

\def\@evenhead{\rlap{\thepage}\hfill{\leftmark}\hfill}
\def\@oddhead{\hfill{\rightmark}\hfill\llap{\thepage}}

\def\chaptermark#1
{\markboth{\normalfont\footnotesize\MakeTextUppercase{\chaptername~\thechapter\enspace#1}}
{\normalfont\footnotesize\MakeTextUppercase{\chaptername~\thechapter\enspace#1}}}

\def\schaptermark#1
{\markboth{\normalfont\footnotesize\MakeTextUppercase{#1}}
{\normalfont\footnotesize\MakeTextUppercase{#1}}}

\def\sectionmark#1
{\markright{\normalfont\footnotesize\MakeTextUppercase{\thesection\enspace #1}}}


% Bibliography
\global\renewenvironment{thebibliography}[1]
     {\chapter*{\bibname}
      \list{\@biblabel{\@arabic\c@enumiv}}
           {\settowidth\labelwidth{\@biblabel{#1}}
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}
      \endlist}


% Tables and figures

\long\def\@makecaption#1#2{
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\small\bf #1\rm\enskip #2}
  \ifdim \wd\@tempboxa >\hsize
    {\small\bf#1\rm\enskip #2\par}
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}
  \fi
  \vskip\belowcaptionskip}


% Mathematics

% Equation numbering
\renewcommand{\theequation}{\rm\thechapter.\arabic{equation}}

% Theorem-like environments
\newtheoremstyle{theo}
  {\topsep}{\topsep}% Space above and below
  {\slshape}% Body style
  {0pt}% Heading indent amount
  {\bfseries}{.}% Heading font and punctuation after it
  {1ex plus 0pt minus .2ex}% Space after heading
  {}% Head spec (empty = same as ‘plain’ style
\theoremstyle{definition}
\newtheorem{Def}{\@defname}[chapter]
\theoremstyle{theo}
\newtheorem{Axi}{\@axiomname}[chapter]
\newtheorem{Conj}{\@conjecturename}[chapter]
\newtheorem{Lem}{\@lemmaname}[chapter]
\newtheorem{Theo}{\@theoname}[chapter]
\newtheorem{Prop}{\@propname}[chapter]
\newtheorem{Cor}{\@corname}[chapter]
%\renewcommand{\qedsymbol}{\rule{3pt}{8pt}}
%\renrewcommand{\proofname}{\@proofname}


% Reference macros

\newcommand{\figref}[2][]{\@figurename~\ref{#2}#1\xspace}
\newcommand{\tabref}[1]{\@tablename~\ref{#1}\xspace}
\newcommand{\eqnref}[1]{\@equationame~\eqref{#1}\xspace}
\newcommand{\chapref}[1]{\@chaptername~\ref{#1}\xspace}
\newcommand{\secref}[1]{\@sectionname~\ref{#1}\xspace}
\newcommand{\appref}[1]{\@appendixname~\ref{#1}\xspace}
\newcommand{\axiref}[1]{\@axiomname~\ref{#1}\xspace}
\newcommand{\conjref}[1]{\@conjname~\ref{#1}\xspace}
\newcommand{\defref}[1]{\@defname~\ref{#1}\xspace}
\newcommand{\lemref}[1]{\@lemmaname~\ref{#1}\xspace}
\newcommand{\theoref}[1]{\@theoname~\ref{#1}\xspace}
\newcommand{\corref}[1]{\@corname~\ref{#1}\xspace}
\newcommand{\propref}[1]{\@propname~\ref{#1}\xspace}
\newcommand{\pgref}[1]{\@pagename~\pageref{#1}\xspace}
%% 
%% This is file `sciposter.cls' v 2.0
%%
%% Authors Michael H.F. Wilkinson and Alle Meije Wink
%%   September 11, 2006
%%    
%% Derived from IWIposter.cls, 18/11/2003, same authors 
%% Problems, bugs and comments to 
%%        m.h.f.wilkinson@rug.nl
%% version 2.0,  2006/09/08 : MAJOR OVERHAUL
%%               - Paper and text area dimension settings made compatible 
%%                 with textpos.sty and wallpaper.sty
%%               - Fancy headers (fancyhdr.sty) now properly supported
%%               - Fonts no longer set using times.sty
%%               - a0size.sty no longer needed
%%               - changed font size setting (using sciposterXX.clo files)
%%                 and 43pt option added.
%%               - \veryHuge, \VeryHuge, and \VERYHuge sizes properly
%%                 implemented for all font sizes (14-43pt)
%%               - change in paper<type>.cfg files: DO NOT use older paper
%%                 definition files
%%               - ISO A-1 support added through papera-1.cfg
%%               - spacing around list environments improved 
%%                 (adapts to font size)
%%                
%% version 1.18, 2006/08/18 :
%%               - footer layout options changed \LEFTSIDEfootlogo command
%%                 added
%%               - \PARstart command problem fixed, by redefining it using 
%%                 the lettrine package by Daniel Flipo
%%               - bug in babel support fixed (\abstractname hook used now)
%% version 1.17, 2006/06/15 :
%%               - interfacing to babel implemented by using \figurename 
%%                 and \tablename in table caption. Thanks to Marcus
%%                 Guenther for pointing out the problem.
%%               - \algorithmname added for easy changing of algorithm 
%%                 caption start, for similar reasons as above
%%               - posibilities for changing caption style added
%%               - \setmargins added as command
%%               - \noindent added to figure, algorithm, and table 
%%                 environments to obtain correct layout if \parindent 
%%                 is not zero (the default). Suggested by Pavel Rehulka
%% version 1.16, 2006/03/28 : 
%%               - plainboxsection option added, by popular request
%% version 1.15, 2005/10/20 : 
%%               - \caption given extra optional parameter indicating
%%                 which fraction of \columnwidth the width should be
%%                 (default 1). Thanks to Andreas Engel for this 
%%                 suggestion.
%% version 1.14, 2005/09/02 :
%%               - \centering (in stead of the center environment) can now 
%%                 be used within figure environment without unwanted 
%%                 centering of the caption
%%               - Bug in colour setting of \section* fixed
%%               - Font pointsize options now also work properly with 
%%                 external paper size settings (including ISO RA0-RA3, 
%%                 ISO B0-B3, and custom).
%% version 1.13, 2005/07/13 :
%%               - mainCol now properly affects background \pagecolor, 
%%                 even in the presence of a right logo
%% version 1.12, 2004/10/11 : 
%%               - mainCol now properly affects background \pagecolor
%%
%% version 1.11, 2004/06/22 :
%%               - Some cosmetics of the \subfigure command improved, 
%%                 references to subfigures are now displayed as 2(a) etc.
%% version 1.10, 2004/04/02 : 
%%               - Standard LaTeX seems to work now: xdvi look weird,
%%                 but with up-to-date dvips I get good results when viewed
%%                 with gv
%%               - Custom paper support through external files added 
%%                 ISO paper sizes RA0, RA1, RA2, B1, B2, and B3 added
%%                 using this feature
%%               - ANSI paper sizes B (tabloid), C, D, and E added
%%               - oddball paper size A0b removed (was this an attempt at
%%                 ANSI E?).
%%               - Boxed section headers work better on small paper now.
%% version 1.00, 2004/02/16 : 
%%               - Essentially IWIposter.cls v1.16, but without explicit 
%%                 reference to University of Groningen and Institute for
%%                 Mathematics and Computing Science
%%                 Formatting for logos is provided but no defaults 
%%                 available
%% 
\ProvidesClass{sciposter}[2006/09/07 v2.0 scientific poster class (MHFW, AMW)]
\NeedsTeXFormat{LaTeX2e}[1995/06/01]
\LoadClass{article}
\RequirePackage{ifthen}
\RequirePackage{lettrine}

\newcommand{\papertype}{a0}
\newcommand{\requestpointsize}{25}
\newcommand{\fontpointsize}{25}
\newlength{\papermargin}
\newlength{\vmargin}
\newlength{\hmargin}


\newcommand{\setpspagesize}{% for default size A0
  \ifthenelse{\equal{\orientation}{portrait}}{
    \special{papersize=83.96cm,118.82cm}
    }{\special{papersize=118.82cm,83.96cm}
    }
  }

\newcommand{\sectionsize}{\normalsize}
\newcommand{\subsectionsize}{\large}

\newboolean{pointsizeset}

\newboolean{largeletters}

\newboolean{userightlogo}
\newboolean{useleftlogo}

\newcommand{\orientation}{portrait}
\newcommand{\printstyle}{final}
\newcommand{\secstyle}{boxed}

\DeclareOption{largefonts}{
\setboolean{largeletters}{true}
}

\DeclareOption{boxedsections}{
\renewcommand{\secstyle}{boxed}
}
\DeclareOption{ruledsections}{
\renewcommand{\secstyle}{ruled}
\renewcommand{\sectionsize}{\Large}
\renewcommand{\subsectionsize}{\large}
}
\DeclareOption{plainsections}{
\renewcommand{\secstyle}{plain}
\renewcommand{\sectionsize}{\Large}
\renewcommand{\subsectionsize}{\large}
}
\DeclareOption{plainboxedsections}{
\renewcommand{\secstyle}{plainboxed}
}

% BUILT IN ISO paper size support
\DeclareOption{a0}{% ISO A0 paper
  \renewcommand{\papertype}{a0}
  \renewcommand{\fontpointsize}{25}
  \setlength{\paperwidth}{83.96cm}
  \setlength{\paperheight}{118.82cm}
  \renewcommand{\setpspagesize}{% not really necessary
    \ifthenelse{\equal{\orientation}{portrait}}{
      \special{papersize=83.96cm,118.82cm}
      }{\special{papersize=118.82cm,83.96cm}
      }
    }
  }
\DeclareOption{a1}{% ISO A1 paper
  \renewcommand{\papertype}{a1}
  \renewcommand{\fontpointsize}{20}
  \setlength{\paperwidth}{59.4cm}
  \setlength{\paperheight}{83.96cm}
  \renewcommand{\setpspagesize}{
    \ifthenelse{\equal{\orientation}{portrait}}{
      \special{papersize=59.4cm,83.96cm}
      }{\special{papersize=83.96cm,59.4cm}
      }
    }
  }
\DeclareOption{a2}{% ISO A2 paper
  \renewcommand{\papertype}{a2}
  \renewcommand{\fontpointsize}{17}
  \setlength{\paperwidth}{41.98cm}
  \setlength{\paperheight}{59.4cm}
  \renewcommand{\setpspagesize}{
    \ifthenelse{\equal{\orientation}{portrait}}{
      \special{papersize=41.98cm,59.4cm}
      }{\special{papersize=59.4cm,41.98cm}
      }
    }
  }
\DeclareOption{a3}{% ISO A3 
  \renewcommand{\papertype}{a3}
  \renewcommand{\fontpointsize}{14}
  \setlength{\paperwidth}{29.7cm}
  \setlength{\paperheight}{41.98cm}
  \renewcommand{\setpspagesize}{
    \ifthenelse{\equal{\orientation}{portrait}}{
      \special{papersize=29.7cm,41.98cm}
      }{\special{papersize=41.98cm,29.7cm}
      }
    }
  }
%%%%% New Options for US page sizes
\DeclareOption{ansiE}{% ANSI E, 34"x44"
  \renewcommand{\papertype}{ansiE}
  \renewcommand{\fontpointsize}{25}
  \setlength{\paperwidth}{86.36cm}
  \setlength{\paperheight}{111.76cm}
  \renewcommand{\setpspagesize}{
    \ifthenelse{\equal{\orientation}{portrait}}{
      \special{papersize=86.36cm,111.76cm}
      }{\special{papersize=111.76cm,86.36cm}
      }
    }
}
\DeclareOption{ansiD}{% ANSI D, 22"x34"
  \renewcommand{\papertype}{ansiD}
  \renewcommand{\fontpointsize}{20}
  \setlength{\paperwidth}{55.88cm}
  \setlength{\paperheight}{86.36cm}
  \renewcommand{\setpspagesize}{
    \ifthenelse{\equal{\orientation}{portrait}}{
      \special{papersize=55.88cm,86.36cm}
      }{\special{papersize=86.36cm,55.88cm}
      }
    }
}
\DeclareOption{ansiC}{% ANSI C, 17"x22"
  \renewcommand{\papertype}{ansiC}
  \renewcommand{\fontpointsize}{17}
  \setlength{\paperwidth}{43.18cm}
  \setlength{\paperheight}{55.88cm}
  \renewcommand{\setpspagesize}{
    \ifthenelse{\equal{\orientation}{portrait}}{
      \special{papersize=43.18cm,55.88cm}
      }{\special{papersize=55.88cm,43.18cm}
      }
    }
}
\DeclareOption{tabloid}{% Tabloid AKA ledger AKA ANSI B, 11"x17"
  \renewcommand{\papertype}{tabloid}
  \renewcommand{\fontpointsize}{14}
  \setlength{\paperwidth}{27.94cm}
  \setlength{\paperheight}{43.18cm}
  \renewcommand{\setpspagesize}{
    \ifthenelse{\equal{\orientation}{portrait}}{
      \special{papersize=27.94cm,43.18cm}
      }{\special{papersize=43.18cm,27.94cm}
      }
    }
}
\DeclareOption{landscape}{
\renewcommand{\orientation}{landscape}
}
\DeclareOption{portrait}{
\renewcommand{\orientation}{portrait}
}
\DeclareOption{draft}{
\renewcommand{\printstyle}{draft}
}
\DeclareOption{final}{
\renewcommand{\printstyle}{final}
}
\DeclareOption{43pt}{
\renewcommand{\requestpointsize}{43}
\setboolean{pointsizeset}{true}
}
\DeclareOption{36pt}{
\renewcommand{\requestpointsize}{36}
\setboolean{pointsizeset}{true}
}
\DeclareOption{30pt}{
\renewcommand{\requestpointsize}{30}
\setboolean{pointsizeset}{true}
}
\DeclareOption{25pt}{
\renewcommand{\requestpointsize}{25}
\setboolean{pointsizeset}{true}
}
\DeclareOption{20pt}{
\renewcommand{\requestpointsize}{20}
\setboolean{pointsizeset}{true}
}
\DeclareOption{17pt}{
\renewcommand{\requestpointsize}{17}
\setboolean{pointsizeset}{true}
}
\DeclareOption{14pt}{
\renewcommand{\requestpointsize}{14}
\setboolean{pointsizeset}{true}
}


\DeclareOption*{\InputIfFileExists{paper\CurrentOption.cfg}{}{\PackageWarning{sciposter}{Unknown option \CurrentOption}}}

\ExecuteOptions{portrait,a0,final}
\ProcessOptions\relax


\ifthenelse{\equal{\orientation}{landscape}}{
  \newlength{\tempsize}
  \setlength{\tempsize}{\paperwidth}
  \setlength{\paperwidth}{\paperheight}
  \setlength{\paperheight}{\tempsize}
  }{}


\ifthenelse{\boolean{pointsizeset}}{
  \renewcommand{\fontpointsize}{\requestpointsize}}{
  \ifthenelse{\boolean{largeletters}}{
    \ifthenelse{\equal{\fontpointsize}{20}}{
      \renewcommand{\fontpointsize}{25}
    }{ \ifthenelse{\equal{\fontpointsize}{17}}{
        \renewcommand{\fontpointsize}{20}
      }{ \ifthenelse{\equal{\fontpointsize}{14}}{
          \renewcommand{\fontpointsize}{17}
        }{\ifthenelse{\equal{\fontpointsize}{25}}{
            \renewcommand{\fontpointsize}{30}
          }{\ifthenelse{\equal{\fontpointsize}{30}}{
              \renewcommand{\fontpointsize}{36}
            }{\renewcommand{\fontpointsize}{43}
            }
          }
        }
      }
    }
  }
}


\RequirePackage{epsfig}
\RequirePackage{graphics}
\RequirePackage{color}
\RequirePackage{shadow}
\RequirePackage{exscale}
%\RequirePackage{mathptmx}
%\RequirePackage{scpmathsize}
%\RequirePackage{courier}
%\RequirePackage[scaled]{helvet}
\RequirePackage{boxedminipage}


\newlength{\titlewidth}
\newlength{\logowidth}

\newcommand{\setbothmargins}[2]{%
  \setlength{\hmargin}{#1}
  \setlength{\vmargin}{#2}

  \setlength{\textwidth}{\paperwidth}
  \addtolength{\textwidth}{-2\hmargin}
  \setlength{\textheight}{\paperheight}
  \addtolength{\textheight}{-2\vmargin}
  
  \setlength{\headheight}{\fontpointsize pt}
  \setlength{\headsep}{0.65\vmargin}
  \addtolength{\headsep}{-\headheight}

  \setlength{\topmargin}{-1in}
  \addtolength{\topmargin}{\vmargin}
  \addtolength{\topmargin}{-\headsep}
  \addtolength{\topmargin}{-\headheight}

  \setlength{\oddsidemargin}{-1in}
  \addtolength{\oddsidemargin}{\hmargin}
  
  \setlength{\footskip}{0.65\vmargin}

  \ifthenelse{\equal{\orientation}{landscape}}{
    \setlength{\logowidth}{0.12\textheight}
    }{
    \setlength{\logowidth}{0.12\textwidth}
    }
  \setlength{\titlewidth}{\textwidth}
  \addtolength{\titlewidth}{-2\logowidth}  
}

\newcommand{\setmargins}[1][]{
  \ifthenelse{\equal{#1}{}}{
    \ifthenelse{\equal{\orientation}{landscape}}{
      \setlength{\papermargin}{0.06\paperheight}
    }{
      \setlength{\papermargin}{0.06\paperwidth}
    }
  }{\setlength{\papermargin}{#1}}
  \setbothmargins{\papermargin}{\papermargin}
}

\setmargins

\InputIfFileExists{sciposter\fontpointsize.clo}{%
  \PackageInfo{sciposter}{Font size successfully set to \fontpointsize pt}
}{
  \PackageWarning{sciposter}{File sciposter\fontpointsize.clo not found, defaulting to 25 pt}

  \renewcommand{\tiny}{\fontsize{12}{14}\selectfont}
  \renewcommand{\scriptsize}{\fontsize{14.4}{18}\selectfont}   
  \renewcommand{\footnotesize}{\fontsize{17.28}{22}\selectfont}
  \renewcommand{\small}{\fontsize{20.74}{25}\selectfont}
  \renewcommand{\normalsize}{\fontsize{24.88}{30}\selectfont}
  \renewcommand{\large}{\fontsize{29.86}{37}\selectfont}
  \renewcommand{\Large}{\fontsize{35.83}{45}\selectfont}
  \renewcommand{\LARGE}{\fontsize{43}{54}\selectfont}
  \renewcommand{\huge}{\fontsize{51.6}{64}\selectfont}
  \renewcommand{\Huge}{\fontsize{61.92}{77}\selectfont}
  \newcommand{\veryHuge}{\fontsize{74.3}{93}\selectfont}
  \newcommand{\VeryHuge}{\fontsize{89.16}{212}\selectfont}
  \newcommand{\VERYHuge}{\fontsize{107}{134}\selectfont}
  
  \newcommand{\titlesize}{\VeryHuge}
  \newcommand{\authorsize}{\LARGE}
  \newcommand{\instsize}{\Large}
  
  \setlength\smallskipamount{6pt plus 2pt minus 2pt}
  \setlength\medskipamount{12pt plus 4pt minus 4pt}
  \setlength\bigskipamount{24pt plus 8pt minus 8pt}
  \setlength\abovecaptionskip{25pt}
  \setlength\belowcaptionskip{0pt}
  \setlength\abovedisplayskip{25pt plus 6pt minus 15 pt}
  \setlength\abovedisplayshortskip{0pt plus 6pt}
  \setlength\belowdisplayshortskip{13pt plus 7pt minus 6pt}
    
}

%% 
%% This is part is adapted from file `a0size.sty' (v1.21b) included 
%% in the a0poster package
%% 
%% Copyright (C) 07.05.97 Gerlinde Kettl and Matthias Weiser
%% Modified to include larger font sizes, and be more future proof

\ProvidesFile{scpmathsize.sty}[2006/09/08 v1.0 CM font size declarations]
 
\DeclareFontShape{OT1}{cmr}{m}{n}
   {  <5> <6> <7> <8> <9> <10> <12> gen * cmr
      <10.95> cmr10
      <14.4>  cmr12
      <17.28->cmr17}{}

\DeclareFontShape{OT1}{cmr}{m}{sl}
    {
      <5><6><7>cmsl8%
      <8> <9> gen * cmsl
      <10><10.95>cmsl10%
      <12-> cmsl12}{}

\DeclareFontShape{OT1}{cmr}{m}{it}
    {
      <5-7>cmti7%
      <8>cmti8%
      <9>cmti9%
      <10-10.95>cmti10%
      <12-> cmti12}{}

\DeclareFontShape{OT1}{cmr}{m}{sc}
    {
      <-> cmcsc10}{}

\DeclareFontShape{OT1}{cmr}{m}{ui}
   {
      <-> cmu10}{}
      
\DeclareFontShape{OT1}{cmr}{b}{n}
     {
      <-> cmb10}{}

\DeclareFontShape{OT1}{cmr}{bx}{n}
   {
      <5> <6> <7> <8> <9> gen * cmbx
      <10><10.95> cmbx10
      <12-> cmbx12}{}
       
\DeclareFontShape{OT1}{cmr}{bx}{sl}
      {
      <5-> cmbxsl10}{}

\DeclareFontShape{OT1}{cmr}{bx}{it}
      {
      <-> cmbxti10}{}

\DeclareFontShape{OT1}{cmss}{m}{n}
     {
      <5-8>cmss8
      <9>cmss9
      <10-10.95>cmss10
      <12-14.4>cmss12
      <17.28-> cmss17}{}

\DeclareFontShape{OT1}{cmss}{m}{sl}
    {
      <5-8>cmssi8 <9> cmssi9
      <10><10.95>cmssi10
      <12-14.4>cmssi12
      <17.28-> cmssi17}{}

\DeclareFontShape{OT1}{cmss}{sbc}{n}
     {
      <-> cmssdc10 }{}

\DeclareFontShape{OT1}{cmss}{bx}{n}
     {
      <-> cmssbx10}{}

\DeclareFontShape{OT1}{cmtt}{m}{n}
     {
      <5-8>cmtt8 <9> cmtt9
      <10><10.95>cmtt10
      <12-> cmtt12}{}
      
\DeclareFontShape{OT1}{cmtt}{m}{it}
     {
      <-> cmitt10}{}     
     
\DeclareFontShape{OT1}{cmtt}{m}{sl}
     {
      <-> cmsltt10}{}
     
\DeclareFontShape{OT1}{cmtt}{m}{sc}
     {
      <-> cmtcsc10}{}     

\DeclareFontShape{OML}{cmm}{m}{it}
    { <5> <6> <7> <8> <9> gen * cmmi
      <10><10.95>cmmi10
      <12-> cmmi12}{}

\DeclareFontShape{OML}{cmm}{b}{it}{%
      <5> <6> <7> <8> <9> gen * cmmib
      <10-> cmmib10}{}


\DeclareFontShape{OMS}{cmsy}{m}{n}{%
      <5> <6> <7> <8> <9> gen * cmsy
      <10-> cmsy10}{}

\DeclareFontShape{OMS}{cmsy}{b}{n}{%
      <5> <6> <7> <8> <9> gen * cmbsy
      <10-> cmbsy10}{}

\DeclareFontShape{OMX}{cmex}{m}{n}{%
      <5-7>sfixed * cmex7
      <8> cmex8
      <9> cmex9
      <10-> cmex10}{}

\DeclareFontShape{U}{cmr}{m}{n}
   {  <5> <6> <7> <8> <9> <10> <12> gen * cmr
      <10.95> cmr10
      <14.4>  cmr12
      <17.28-> cmr17}{}

\DeclareFontShape{U}{cmr}{m}{sl}
    {
      <5-7>cmsl8%
      <8> <9> gen * cmsl
      <10><10.95>cmsl10%
      <12-> cmsl12}{}

\DeclareFontShape{U}{cmr}{m}{it}
    {
      <5-7>cmti7%
      <8>cmti8%
      <9>cmti9%
      <10><10.95>cmti10%
      <12-> cmti12}{}

\DeclareFontShape{U}{cmr}{m}{sc}
    {
      <-> cmcsc10}{}

\DeclareFontShape{U}{cmr}{m}{ui}
   {
      <-> cmu10}{}

\DeclareFontShape{U}{cmr}{b}{n}
     {
      <-> cmb10}{}

\DeclareFontShape{U}{cmr}{bx}{n}
   {
      <5> <6> <7> <8> <9> gen * cmbx
      <10><10.95> cmbx10
      <12-> cmbx12}{}

\DeclareFontShape{U}{cmr}{bx}{sl}
      {
      <-> cmbxsl10}{}

\DeclareFontShape{U}{cmr}{bx}{it}
      {
      <-> cmbxti10}{}


\DeclareMathSizes{24.88}{24.88}{20.74}{14.4}
\DeclareMathSizes{29.86}{29.86}{20.74}{14.4}
\DeclareMathSizes{35.83}{35.83}{24.88}{17.28}
\DeclareMathSizes{43.00}{43.00}{35.83}{24.88}
\DeclareMathSizes{51.6}{51.6}{35.83}{24.88}
\DeclareMathSizes{61.92}{61.92}{43}{29.86}
\DeclareMathSizes{74.3}{74.3}{51.6}{35.83}
\DeclareMathSizes{89.16}{89.16}{61.92}{43}
\DeclareMathSizes{107}{107}{74.3}{51.6}
\DeclareMathSizes{128.4}{128.4}{89.16}{61.92}
\DeclareMathSizes{154.08}{154.08}{107}{74.3}
\DeclareMathSizes{184.9}{184.9}{128.4}{89.16}

% end of inheritance from a0size.sty

\setlength\belowdisplayskip\abovedisplayskip
\normalsize
\newlength{\parstartskip}
\setlength\parstartskip{2\baselineskip}
\setboolean{useleftlogo}{true}
\setboolean{userightlogo}{true}

% Settings for list environments added

\setlength\leftmargini{1.25em}
\setlength\leftmarginii{0.9em}
\setlength\leftmarginiii{0.8em}
\setlength\leftmarginiv{0.8em}
\setlength\leftmarginv{0.8em}
\setlength\leftmarginvi{0.8em}
\setlength\leftmargin\leftmargini
\setlength\labelsep{.25em}
\setlength\labelwidth\leftmargini\advance\labelwidth-\labelsep



% Definition of Colors
% Background- and Text-color
\definecolor{mainCol}{rgb}{1,1,1}
\definecolor{BoxCol}{rgb}{0.9,0.9,0.9}
\definecolor{TextCol}{rgb}{0,0,0}
\definecolor{SectionCol}{rgb}{0,0,0}


\AtBeginDocument{
  \typeout{-- Paper type set to \papertype.}

 \@ifundefined{pdfoutput}{% standard LaTeX
    \setpspagesize
    }{}
  \pagestyle{empty}
  \setfooter
  \pagecolor{mainCol} 
  }
%


\newlength{\figbotskip}
\newlength{\figtopskip}
\newlength{\secskip}
\newlength{\titleskip}
\ifthenelse{\equal{\orientation}{landscape}}{
  \setlength{\titleskip}{0.025\textheight}
}{
  \setlength{\titleskip}{0.025\textwidth}
}
\setlength{\secskip}{\bigskipamount}
\setlength{\figtopskip}{\secskip}
\setlength{\figbotskip}{\secskip}
%%%% Defines default institute
\newcommand{\theinstitute}{%
}

\newcommand{\printemail}{}
%
\newcommand{\printleftlogo}{%
  \begin{center}
    \hspace*{0.6\textwidth}
  \end{center}
}
\newcommand{\printrightlogo}{%
  \begin{center}
    \hspace*{0.6\textwidth}
  \end{center}
}
%
%
%
\newboolean{footlogo@leftside}
%
\newcommand{\LEFTSIDEfootlogo}[1][true]{%
  \setboolean{footlogo@leftside}{#1}
  \setfooter
} 
%
\newcommand{\footlogo}{}
\newcommand{\@conference}{}
%
\newcommand{\setfooter}{%
  \ifthenelse{\boolean{footlogo@leftside}}{%
    \renewcommand{\@oddfoot}{\footlogo \hfill \@conference\\}
  }{%
    \renewcommand{\@oddfoot}{\@conference \hfill \footlogo\\}
  }
}
%
%
\newcommand{\conference}[1]{% arg:name of conference 
\makeatletter
\renewcommand{\@conference}{\footnotesize #1}
\setfooter
}

\newlength{\authorwidth}
\setlength{\authorwidth}{\titlewidth}

\newcommand{\affil}[1]{\instsize \normalfont #1}

\newcommand{\authoralign}{t}


\renewcommand{\author}[1]{%
  \gdef\@author{\begin{minipage}[\authoralign]{\authorwidth}%
      \centering #1 
      \end{minipage}
    }    
}

\renewcommand{\maketitle}{%
  \def\and{\end{minipage} \hspace{1em} \begin{minipage}[\authoralign]{\authorwidth}\centering }%
  \ifthenelse{\boolean{useleftlogo}}{%
  \begin{minipage}[c]{\logowidth}
     \printleftlogo
  \end{minipage}
  }{}
  \begin{minipage}[c]{\titlewidth}
    \begin{center}
      %\renewcommand{\baselinestretch}{3.3}\normalsize
      {\titlesize \bf \@title}\par
      %\renewcommand{\baselinestretch}{1.0}\normalsize            
      \vspace{0.6\titleskip}
      {\authorsize {\bf \@author}}
      {\par
       \instsize
        \theinstitute \par
       \ifthenelse{\equal{\printemail}{}}{%nothing
         }{%
         \vspace{0.2\titleskip}
         \instsize
         \texttt{\printemail}
         }
      }
    \end{center}
  \end{minipage}%}}
  \ifthenelse{\boolean{userightlogo}}{%
  \begin{minipage}[c]{\logowidth}
     \printrightlogo
  \end{minipage}
  }{}
\vspace{\titleskip}
}
\newlength{\secboxwidth}
%%% \replacement for \section
% with colored section-title and automatic numbering
\newcommand{\Section}[2][?]{
  \vspace{\secskip}
  \refstepcounter{section}
  \begin{center}
    \setlength{\secboxwidth}{\columnwidth}
    \addtolength{\secboxwidth}{-1cm}
    \shabox{\colorbox{BoxCol}{
      \begin{minipage}[c]{\secboxwidth}
        \vspace{1.1ex}
        \begin{center}
          {\bf\sectionsize\textcolor{SectionCol}{\arabic{section}.~#2}}
        \end{center}
        \vspace{0.2ex}
      \end{minipage}
    }}
  \end{center}
  \par\vspace{0.375\secskip}
}
%
\newcommand{\sSection}[1]{
  \vspace{\secskip}
  \begin{center}
    \setlength{\secboxwidth}{\columnwidth}
    \addtolength{\secboxwidth}{-1cm}
    \shabox{\colorbox{BoxCol}{
      \begin{minipage}[c]{\secboxwidth}
        \vspace{1.1ex}
        \begin{center}
          {\bf\sectionsize\textcolor{SectionCol}{#1}}
        \end{center}
        \vspace{0.2ex}
      \end{minipage}
    }}
  \end{center}
  \par\vspace{0.375\secskip}
}
%
%
%%% \PlainSection - replacement for \section
\newcommand{\PlainSection}[2][?]{
  \refstepcounter{section}
  \vspace{\secskip}
  \begin{center}
  {\bf\sectionsize\textcolor{SectionCol}{\arabic{section}.~#2}}
  \end{center}
  \par\vspace{0.375\secskip}
}
%
%%% \sPlainSection - replacement for \section
\newcommand{\sPlainSection}[1]{
  \vspace{\secskip}
  \begin{center}
  {\bf \sectionsize \textcolor{SectionCol}{#1}}
  \end{center}
  \par\vspace{0.375\secskip}
}
%
%%% \RuledSection - replacement for \section
\newcommand{\RuledSection}[2][?]{
  \vspace{0.3\secskip}
  \rule{\columnwidth}{\secrulewidth}
  \refstepcounter{section}
  \vspace{0.7\secskip}
  \begin{center}
  {\bf\sectionsize\textcolor{SectionCol}{\arabic{section}.~#2}}
  \end{center}
  \par\vspace{0.375\secskip}
}
%
%%% \sRuledSection - replacement for \section
\newcommand{\sRuledSection}[1]{
  \ifthenelse{\equal{\arabic{section}}{0}}{}{
     \vspace{0.3\secskip}
     \rule{\columnwidth}{\secrulewidth}
  }
  \vspace{0.7\secskip}
  \begin{center}
  {\bf \sectionsize \textcolor{SectionCol}{#1}}
  \end{center}
  \par\vspace{0.375\secskip}
}
%%% \replacement for \section
% with colored section-title and automatic numbering
\newcommand{\PlainBoxSection}[2][?]{
  \vspace{\secskip}
  \refstepcounter{section}
  \begin{center}
    \setlength{\secboxwidth}{\columnwidth}
    \addtolength{\secboxwidth}{-1cm}
    \setlength{\fboxrule}{2pt}
    \setlength{\fboxsep}{0pt}
    \fbox{\colorbox{BoxCol}{
      \begin{minipage}[c]{\secboxwidth}
        \vspace{1.1ex}
        \begin{center}
          {\bf\sectionsize\textcolor{SectionCol}{\arabic{section}.~#2}}
        \end{center}
        \vspace{0.2ex}
      \end{minipage}
    }}
  \end{center}
  \par\vspace{0.375\secskip}
}
%
\newcommand{\sPlainBoxSection}[1]{
  \vspace{\secskip}
  \begin{center}
    \setlength{\secboxwidth}{\columnwidth}
    \addtolength{\secboxwidth}{-1cm}
    \setlength{\fboxrule}{2pt}
    \setlength{\fboxsep}{0pt}
    \fbox{\colorbox{BoxCol}{
      \begin{minipage}[c]{\secboxwidth}
        \vspace{1.1ex}
        \begin{center}
          {\bf\sectionsize\textcolor{SectionCol}{#1}}
        \end{center}
        \vspace{0.2ex}
      \end{minipage}
    }}
  \end{center}
  \par\vspace{0.375\secskip}
}
%
%
%
%
%
\ifthenelse{\equal{\secstyle}{ruled}}{
  \renewcommand{\section}{\secdef\RuledSection\sRuledSection}
}{
  \ifthenelse{\equal{\secstyle}{plain}}{
    \renewcommand{\section}{\secdef\PlainSection\sPlainSection}
  }{  
    \ifthenelse{\equal{\secstyle}{plainboxed}}{
      \renewcommand{\section}{\secdef\PlainBoxSection\sPlainBoxSection}
    }{
      \renewcommand{\section}{\secdef\Section\sSection}
    }
  }
}
%%% \SubSection - replacement for \subsection
\newcommand{\SubSection}[2][?]{
  \vspace{0.5\secskip}
  \refstepcounter{subsection}
  {\bf \subsectionsize \textcolor{TextCol}{\arabic{section}.\arabic{subsection}~#2}}
  \par\vspace{0.375\secskip}
}
%%% \sSubSection - replacement for \subsection
\newcommand{\sSubSection}[1]{
  \vspace{0.5\secskip}
  {\bf \subsectionsize \textcolor{TextCol}{#1}}
  \par\vspace{0.375\secskip}
}
%
\renewcommand{\subsection}{\secdef\SubSection\sSubSection}
%%%
\renewenvironment{abstract}
{\section*{\abstractname}\it}
{}
%%%
%%%
\newcounter{subfig}[figure]
\renewcommand{\thesubfig}{\thefigure(\alph{subfig})}
\newcommand{\capstart}{}
\newcommand{\capspace}{\nobreakspace}
\newcommand{\capfirstskip}{}
\newcommand{\caplastskip}{}
\newcommand{\subfigure}[2][]{
\refstepcounter{subfig}
\begin{tabular}{c}
 #2 \\
 (
 \alph{subfig}
 ) #1\\
\end{tabular}
}

%%% caption formatting commands
\newcommand{\mastercapstartstyle}[1]{\textbf{#1}}
\newcommand{\mastercapbodystyle}{\itshape}
\newcommand{\capbodystyle}{\mastercapbodystyle}

\newcommand{\figcapstartstyle}[1]{\mastercapstartstyle{#1}}
\newcommand{\figcapbodystyle}{\mastercapbodystyle}
\newcommand{\tablecapstartstyle}[1]{\mastercapstartstyle{#1}}
\newcommand{\tablecapbodystyle}{\mastercapbodystyle}
\newcommand{\algcapstartstyle}[1]{\mastercapstartstyle{#1}}
\newcommand{\algcapbodystyle}{\mastercapbodystyle}

%%% replacement for standard figure environment
\renewenvironment{figure}[1][h]
{\vspace{\figtopskip}
 \refstepcounter{figure}
 \renewcommand{\capstart}{%
    \figcapstartstyle{\figurename\nobreakspace\thefigure:}%
 }
 \renewcommand{\capfirstskip}{\vspace{0.5\figtopskip}\par}
 \renewcommand{\caplastskip}{}
 \noindent%
 \begin{minipage}{\columnwidth}
 \renewcommand{\capbodystyle}{\figcapbodystyle}
}
{\end{minipage}
\vspace{\figbotskip}}

\newcommand{\algorithmname}{Algorithm}

\newcounter{algorithm}

%%% replacement for algorithm environment
\newenvironment{algorithm}[1][h]
{\vspace{\figtopskip}
 \refstepcounter{algorithm}
 \renewcommand{\capstart}{%
   \algcapstartstyle{\algorithmname\nobreakspace\thealgorithm:}%
 }
 \renewcommand{\capfirstskip}{\vspace{0.25\figtopskip}\par}
 \renewcommand{\caplastskip}{}
 \noindent%
 \setlength{\fboxsep}{0.5em}

 \begin{boxedminipage}{\columnwidth}
 \renewcommand{\capbodystyle}{\algcapbodystyle}
}
{\end{boxedminipage}
\vspace{\figbotskip}}

%%% replacement for standard table environment
\renewenvironment{table}[1][h]
{\vspace{\figtopskip}
 \refstepcounter{table}
 \renewcommand{\capstart}{%
   \tablecapstartstyle{\tablename\nobreakspace\thetable:}%
 }
 \renewcommand{\caplastskip}{\vspace{0.5\figtopskip}\par}
 \renewcommand{\capfirstskip}{}
 \noindent%
 \begin{minipage}{\columnwidth}
 \renewcommand{\capbodystyle}{\tablecapbodystyle}
 }
 {\end{minipage}\vspace{\figbotskip}}
%%% replacement for standard \caption
\renewcommand{\caption}[2][1]{
  \capfirstskip
  \begin{minipage}{#1\columnwidth}
    \renewcommand{\baselinestretch}{1}
    \setbox0=\hbox{\capstart\capspace\capbodystyle #2}
    \ifdim\wd0>\hsize
      {\capstart\capspace\capbodystyle #2}
    \else
      \centerline{\box0}
    \fi
    \par 
  \end{minipage}
  \caplastskip
}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
\setlength{\sboxrule}{2pt}
\setlength{\sboxsep}{0pt}
%
\ifthenelse{\equal{\orientation}{landscape}}{
  \setlength{\columnsep}{0.0385\textheight}
  \setlength{\columnseprule}{0.0025\textheight}
}{
  \setlength{\columnsep}{0.0385\textwidth}
  \setlength{\columnseprule}{0.0025\textwidth}
}
\setlength{\parindent}{0.0cm}
\newlength{\secrulewidth}
\setlength{\secrulewidth}{0.3\columnseprule}

%
%
\renewcommand{\familydefault}{\sfdefault}

\newcommand{\leftlogo}[2][0.9]{%
\ifthenelse{\equal{#2}{}}{%
 \renewcommand{\printleftlogo}{%
   \begin{center}
     \hspace*{0.6\textwidth}
   \end{center}
   }
 }
 {\renewcommand{\printleftlogo}{%
      \begin{center}
      \resizebox{#1\textwidth}{!}{\includegraphics{#2}}
    \end{center}
 }}
}
%
\newcommand{\rightlogo}[2][0.9]{%
\ifthenelse{\equal{#2}{}}{%
 \renewcommand{\printrightlogo}{%
   \begin{center}
     \hspace*{0.6\textwidth}
   \end{center}
   }
 }
 {\renewcommand{\printrightlogo}{%
     \begin{center}
       \resizebox{#1\textwidth}{!}{\includegraphics{#2}}
     \end{center}
 }}
}
%
\newcommand{\email}[1]{%
\renewcommand{\printemail}{#1}
}
%
\newcommand{\institute}[1]{%
\renewcommand{\theinstitute}{#1}
}
%
\newcommand{\norightlogo}{
\setboolean{userightlogo}{false}
\setlength{\titlewidth}{\textwidth}
\ifthenelse{\boolean{useleftlogo}}{
\addtolength{\titlewidth}{-\logowidth}
}{}
}
\newcommand{\noleftlogo}{
\setboolean{useleftlogo}{false}
\setlength{\titlewidth}{\textwidth}
\ifthenelse{\boolean{userightlogo}}{
\addtolength{\titlewidth}{-\logowidth}
}{}
}
\newcommand{\nologos}{
\setboolean{useleftlogo}{false}
\setboolean{userightlogo}{false}
\setlength{\titlewidth}{\textwidth}
}

%
%%% \PARstart command redefined using lettrine package of Danial Flipo
%%  only included here for backwards compatibility 

\newcommand{\PARstart}[2]{\lettrine{#1}{#2}}


\endinput

\color{TextCol}

%% 
%% End of file `sciposter.cls'.











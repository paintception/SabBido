%% 
%% This is file `IWIposter.cls' v 2.03
%%
%% Authors Michael H.F. Wilkinson and Alle Meije Wink
%%   September 8, 2006
%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%                 NOW BOTH for latex and pdflatex use!!!! 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Derived from a0poster.cls, 07.05.97 by Gerlinde Kettl and Matthias Weiser
%% (in the distant past: all their code has been replaced)
%% Problems, bugs and comments to 
%%        michael@cs.rug.nl
%%
%% version 2.03, 2006/09/08 : helvet and courier packages included to adapt to
%%                            to changes in sciposter.cls v2.0
%%
%% version 2.02, 2004/04/27 : Name change of poster.cls to sciposter.cls
%%                            as proposed to CTAN incorporated
%%
%% version 2.01, 2004/04/02 : Double declaration of \footskip (in poster.cls
%%                            and in IWIposter.cls) resolved (now only in
%%                            poster.cls
%% 
%% version 2.00, 2004/02/16 : RADICAL redesign: now based on poster.cls
%%                            (which is IWIposter 1.16 minus logos and 
%%                             institute definitions).
%%
%% version 1.16, 2003/11/18 : \norightlogo, \noleftlogo, and \nologos 
%%                            commands added; \sectionsize and 
%%                            \subsectionsize now control font size
%%                            of (sub)section headers.
%%
%% version 1.15, 2003/11/03 : Font size options 14pt, 17pt, 20pt, 25pt,
%%                            30pt, and 36pt added 
%%
%% version 1.14, 2003/10/28 : RuGbalkBl moved to bottom of page 
%%                            (footer is now always made), right logo 
%%                            is empty by default.
%% 
%% version 1.13, 2003/10/01 : RuGbalkBl replaces RuGbalk as default
%%                            else code inserted for 
%%                            InputIfFileExists{iwilogopath.cfg}
%%                            Poster now pdflatex WITHOUT the file
%%                            BoxCol default set to pale blue
%%
%% version 1.12, 2003/08/13 : iwilogo replaced by RuGbalk as default
%%
%% version 1.11, 2003/06/17 : problems with baselineskip addressed, 
%%                            section headers improved
%%
%% version 1.10, 2003/06/10 : largefonts option implemented
%%
%% version 1.09, 2003/05/13 : iwilogopath.cfg file used to define logo paths
%%                            (also used by iwi.sty)
%%
%% version 1.08, 2003/05/12 : \subfigure command added: do not use
%%                            subfigure package
%%
%% version 1.07, 2003/05/09 : Algorithm environment inserted: do not 
%%                            use algorithm package
%%                            
%%
%% version 1.06, 2003/01/13 : \rightlogo and \leftlogo commands now have 
%%                            width as an optional parameter (0.9 default)
%%
%%
%% version 1.05, 2002/11/13 : Section definitions improved using
%%                            \secdef, so \section* works now
%%                            options added:
%%                                boxedsections
%%                                plainsections
%%                                ruledsections 
%%                            
%%                            these control the section style.
%%                            BoxCol and SectionCol defined to 
%%                            control \section colours
%%                             
%%
%% version 1.04, 2002/10/09 : point sizes for different paper sizes
%%                            defined differently;
%%                            centering of printed matter on page 
%%                            now correct (\hoffset and \voffset set)
%%                            various skips turned into rubber lengths
%%                            for improved formatting
%%
%% version 1.03, 2002/10/08 : paper size dependent part rewritten 
%%                            \footskip problems sorted out
%%     
%% version 1.02, 2002/09/19 : table environment included
%%  
%%
\ProvidesClass{IWIposter}[2006/09/08 v2.03 IWI a0 poster class (MHFW, AMW)]
\NeedsTeXFormat{LaTeX2e}[1995/06/01]

\RequirePackage{ifthen}


%\RequirePackage[scaled]{helvet}
%\renewcommand{\encodingdefault}{T1}
\RequirePackage{calc}
\RequirePackage{settobox}



\newcommand{\usecolor}{color}
\newboolean{usesectionboxes}

\DeclareOption{color}{
\renewcommand{\usecolor}{color}
}

\DeclareOption{bw}{
\renewcommand{\usecolor}{bw}
}

\DeclareOption{sectionbox}{
\setboolean{usesectionboxes}{true}
\PassOptionsToClass{plainsections}{sciposter}
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{sciposter}}
\ProcessOptions\relax

\LoadClass{sciposter}
%\RequirePackage{sectionbox}

\AtBeginDocument{
\pagestyle{empty}

\makeatletter
\setlength{\columnwidth}{\textwidth}

\renewcommand{\@oddfoot}{\hfill \footlogo\\}
}
%
    \newlength{\IWI@temprul}
    \newlength{\tempsep}

%%%% Defines default institute
\institute{ Institute for Mathematics and Computing Science,
University of Groningen, PO Box 800, 9700 AV Groningen, The Netherlands
}
 %%% \replacement for \section
% with colored section-title and automatic numbering
\newcommand{\RUGSection}[2][?]{
  \vspace{\secskip}
  \refstepcounter{section}
  \begin{center}
    \setlength{\secboxwidth}{\columnwidth}
    \addtolength{\secboxwidth}{-0.9cm}
    \setlength{\IWI@temprul}{\fboxrule}
    \setlength{\tempsep}{\fboxsep}
    \setlength{\fboxrule}{0pt}
    \setlength{\fboxsep}{0pt}
    \fbox{\colorbox{BoxCol}{
      \begin{minipage}[c]{\secboxwidth}
        \vspace{1.1ex}
        \begin{center}
          {\bf\sectionsize\textcolor{SectionCol}{\LARGE{\arabic{section}.~#2}}}
        \end{center}
        \vspace{0.2ex}
      \end{minipage}
    }}
    \setlength{\fboxsep}{\tempsep}
    \setlength{\fboxrule}{\IWI@temprul}
  \end{center}
  \par\vspace{0.375\secskip}
}
%
\newcommand{\sRUGSection}[1]{
  \vspace{\secskip}
  \begin{center}
    \setlength{\secboxwidth}{\columnwidth}
    \addtolength{\secboxwidth}{-0.9cm}
    \setlength{\fboxrule}{0pt}
    \setlength{\fboxsep}{0pt}
    \fbox{\colorbox{BoxCol}{
      \begin{minipage}[c]{\secboxwidth}
        \vspace{1.1ex}
        \begin{center}
          {\bf\sectionsize\textcolor{SectionCol}{\LARGE{#1}}}
        \end{center}
        \vspace{0.2ex}
      \end{minipage}
    }}
  \end{center}
  \par\vspace{0.375\secskip}
}


\ifthenelse{\boolean{usesectionboxes}}{}{
\renewcommand{\section}{\secdef\RUGSection\sRUGSection}
}

\definecolor{RUGred}{rgb}{0.797,0,0}

\InputIfFileExists{iwilogopath.cfg}{}{\newcommand{\iwilogopath}{}}
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of Colors
% Background- and Text-color
%
\renewcommand{\author}[1]{%
  \gdef\@author{\begin{minipage}[\authoralign]{\authorwidth}%
     \centering #1 
      \end{minipage}
    }    
}


\setlength{\titlewidth}{\textwidth}
\addtolength{\titlewidth}{-2.5\logowidth}
\addtolength{\titlewidth}{-\tabcolsep}
\setlength{\authorwidth}{\textwidth}
\renewcommand{\maketitle}{%
  \def\and{\end{minipage} \hspace{1em} \begin{minipage}[\authoralign]{\authorwidth}\centering }%


  \begin{tabular}{l r} 
  \setlength{\fboxrule}{0pt}
  \begin{boxedminipage}[t]{2.5\logowidth}
     \printleftlogo
  \end{boxedminipage}
  \begin{boxedminipage}[t]{\titlewidth}
      \raggedleft
      {\titlesize \bf \@title}
  \end{boxedminipage}
  \end{tabular}
  \vspace{0.4\titleskip}
 
  \newsavebox{\authorbox}

  \sbox{\authorbox}{
  \begin{minipage}[b]{\textwidth}
      \centering
      \vspace*{0.4\titleskip}
      {\authorsize {\bf \@author}}

      { \instsize
        \theinstitute \par
       \ifthenelse{\equal{\printemail}{}}{%nothing
         }{%
         \instsize
         \texttt{\printemail}\\[-0.3\titleskip]
         {\tiny \strut}
         }
      }
  \end{minipage}
  }

  \newlength{\authboxheight}
  \settoboxheight{\authboxheight}{\authorbox}

  

  %\makebox[0pt][l]{\large BLA}
  \makebox[0pt][l]{\hspace*{-\papermargin}\includegraphics*[scale=1.5, viewport=0 0
    {0.667\paperwidth} {0.667\authboxheight}, clip=true ]{\iwilogopath RUGhatchA0width.pdf}}
   \usebox{\authorbox}

\vspace{\titleskip}
}

\ifthenelse{\equal{\usecolor}{color}}{
  \ifthenelse{\boolean{usesectionboxes}}{
    \definecolor{SectionCol}{rgb}{0.797,0,0}
    \definecolor{sectboxrulecol}{rgb}{0.797,0,0}
    \definecolor{sectboxfillcol}{gray}{0.95}

  }{
    \definecolor{BoxCol}{rgb}{0.797,0,0}
    \definecolor{SectionCol}{rgb}{1,1,1}
  }
  \leftlogo[0.95]{\iwilogopath RUGR_logoEN_rood_RGB}
}{
  \leftlogo[0.95]{\iwilogopath RUGR_logoEN_zwart_RGB}
}

\setlength{\columnseprule}{0.0pt}


%
%% 
%% End of file `IWIposter.cls'.











function [YResult varargout]=QPmR(Reg,varargin)
% Quasi-Polynomial Mapping based Rootfinder. 
% 
% QPmR(Region,P,D)
%  
%    Finds all zeros of the quasi-polynomial 
%  
%    QP(s)=(P(1,1)*s^n+...+P(1,n)*s+P(1,n+1))*exp(-D(1)*s)+
%     +(P(2,1)*s^n+...+P(2,n)*s+P(2,n+1))*exp(-D(2)*s)+
%                    ...
%     +(P(N,1)*s^n+...+P(N,n)*s+P(N,n+1))*exp(-D(N))*s)
%  
%    located in the complex plane region defined by
%  
%    Region=[real_min real_max imag_min imag_max]
%  
%    The other two function inputs are:
%  
%    P  - N by n matrix of polynomial coefficients, n is the maximum power of s
%         in the quasi-polynomial, N is the number of delays. 
%    D  - vector of different (positive) delay values of size N. One of the 
%         delays should be equal to 0.
%  
%    The quasi-polynomial can also be defined in the function handle form Fun. 
%    Then, the QPmR function syntax is
%  
% QPmR(Region,Fun)
%  
%    The QPmR function output is the vector of all quasi-polynomial zeros located in
%    the given region. The method is based on mapping the quasi-polynomial zero 
%    level curves over the complex plane region with the adaptation of the mapping 
%    grid. 
%  
% NaN as the function output indicates failure of the adaptation algorithm. 
%   In this case, the region Reg should be reduced. Alternatively, the grid can
%   be assigned manually using the extended function modes   
%          
% QPmR(Region,P,D,e,ds,gr) or 
% QPmR(Region,Fun,e,ds,gr)
%  
%   where the additional input parameters are
%  
%    e  - computation accuracy. If e=-1, then e=1e-6*ds (the same accuracy 
%         is considered if e parameter is not given, as it is above).   
%    ds - grid step for mapping the zero-level curves. If ds=-1, the grid 
%         step is adjusted automatically. 
%    gr - graphical representation of the results. If gr=1, results are 
%         visualized in plots (default gr=0). If the quasi-polynomial is 
%         given in P and D, the spectrum distribution diagram and the
%         asymptotic exponentials of the spectra are visualized. If the
%         quasi-polynomial is neutral, also the spectrum of the associated 
%         difference equation is computed and its safe upper bound is 
%         determined and visualized. If gr=1, the information on the grid 
%         adaptation is provided in the command window.
%  
% [R Y]=(Region,P,D,....) provides the following outputs
%  
%    R          - computed zeros of the quasi-polynomial (NaN indicates the
%                 algorithm failure)
%    Y          - structure with summary of the QPmR results, particularly
%    Y.zeros    - computed zeros of the quasi-polynomial (available also 
%                 if R=NaN)
%    Y.flag     - result correctness flag. 
%                 Y.flag=1  - the positive result of cross-checking implies
%                             that the zeros are computed correctly.
%                 Y.flag=0  - method failure: either the region is too 
%                             large or there are multiple or dense 
%                             roots. Next, the function can also be 
%                             ill-conditioned. Try to reduce the
%                             region.
%                 Y.flag=-1 - method failure: too large grid, which causes
%                             Newton's iterations failure. The grid size 
%                             ds should be reduced (manually, if needed). 
%    Y.accuracy - accuracy estimate of the computed zeros
%    Y.asympt   - parameters [mi c] of the asymptotic exponentials of the
%                 root chains real=c(k)-mi(k)*log(imag);
%    Y.function - quasi-polynomial in the function handle form
%    Y.grid     - final grid size 
%  
%    Additional outputs for neutral quasi-polynomials   
%    Y.DEzeros    - computed zeros of the associated difference equation
%    Y.DEflag     - result correctness flag (the same as above) 
%    Y.DEaccuracy - accuracy estimate of the computed zeros
%    Y.DEupbound  - safe upper bound on the spectrum of difference equation
%    Y.DEfunction - difference equation in the function handle form
%    Y.DEgrid     - final grid size
%  
%  
% [R Y]=(Region,Fun,....) provides the following outputs
%  
%    R          - computed zeros of the quasi-polynomial (NaN indicates the
%                 algorithm failure)
%    Y          - summary of the QPmR function results.
%    Y.zeros    - computed zeros of the quasi-polynomial
%    Y.flag     - result correctness flag - the same as above              
%    Y.accuracy - accuracy estimate of the computed zeros
%    Y.grid     - final grid size 
%  
% Remark 1: in the QPmR(Reg,Fun,...), the function can be used for computing 
% roots of general analytical functions, e.g. fractional polynomials or
% quasi-polynomials.
%  
% Remark 2: in the automatic adjustment of the grid size (ds==-1), first, 
% the grid size is set to ds=(Reg(2)-Reg(1))*(Reg(4)-Reg(3))/Ns, 
% where Ns=1000 (or Ns=500 for more complex functions). If not sufficient, 
% the grid size is up to twice time reduced by the factor of four. 
% If not sufficient, the region is divided first to four and then up to 16 
% Sub-regions if needed, and the QPmR runs recursively in two recursion levels.  
%  
%  
% Example
%    Find all the roots of the quasi-polynomial
%    Q(s)=(1.5*s^3+0.2*s^2+20.1)+(s^3-2.1*s)*exp(-s*1.3)+
%         +3.2*s*exp(-s*3.5)+1.4*exp(-s*4.3)
%    located in the region Reg=[-10 5 0 300]
%  
%  
% a) representation of the quasi-polynomial by the coefficient matrix and
% vectors of delays:
%  
% P=[1.5 0.2 0 20.1;1 0 -2.1 0;0 0 3.2 0;0 0 0 1.4]
% D=[0;1.3;3.5;4.3]
%  
% R=QPmR([-10 5 0 300],P,D) - provides the computed zeros in the vector R. 
%    No graphical outputs, accuracy and grid size are adjusted automatically. 
%    
% [R Y]=QPmR([-10 5 0 300],P,D,-1,-1,1) - provides the computed zeros in the
%    vector R. Structure Y contains additional information on the spectrum 
%    and its computational aspects. With graphical outputs, accuracy 
%    and grid size are adjusted automatically.
%  
% [R Y]=QPmR([-10 5 0 300],P,D,1e-8) - with given accuracy 1e-8. Grid size 
%    is adjusted automatically, no graphical outputs.
%  
% [R Y]=QPmR([-10 5 0 300],P,D,1e-8,0.1) - with given accuracy 1e-8 and the 
%    fixed grid size 0.1. No graphical outputs. No grid adaptation.
%  
% b)  representation of the quasi-polynomial by the function handle
%  
%    Fun=@(s)(1.5*s.^3+0.2*s.^2+20.1)+(s.^3-2.1.*s).*exp(-s*1.3)+
%         3.2.*s.*exp(-s*3.5)+1.4.*exp(-s*4.3)
%  
%    R=QPmR([-10 5 0 300],Fun) 
%  
%    [R Y]=QPmR([-10 5 0 300],P,D,-1,-1,1)
% 
% QPmR v.2
% Created by Tomas Vyhlidal, CTU in Prague
% http://www.cak.fs.cvut.cz/algorithms/qpmr


if isnumeric(varargin{1})
    %matrix based formulation of the quasipolynomial
    P=varargin{1};
    Z=varargin{2};
    if nargin==3
        varargin{3}=-1;
        varargin{4}=-1;
        varargin{5}=-1;
    elseif nargin==4
        varargin{4}=-1;
        varargin{5}=-1;
    elseif nargin==5
        varargin{5}=-1;
    end

    
    if varargin{4}==-1;
        if sum(size(P))>15
            ds=(Reg(2)-Reg(1))*(Reg(4)-Reg(3))/500;
        else
            ds=(Reg(2)-Reg(1))*(Reg(4)-Reg(3))/1000;
        end
    else
        ds=varargin{4};
    end
    
    
    if varargin{3}==-1;
        e=ds/1e6;
    else
        e=varargin{3};
    end
    
    %slight extension of the region
    bmin=Reg(1)-3*ds;
    bmax=Reg(2)+3*ds;
    wmin=Reg(3)-3*ds;
    wmax=Reg(4)+3*ds;
    beta=bmin:ds:bmax;
    omega=wmin:ds:wmax;
   
   %mapping the zerolevel curves of the function
   [B W]=meshgrid(beta,omega);
   S=B+j*W;
   rad=length(P(1,:))-1;
   poczp=length(Z);
   for k=1:rad+1;
       Sp(k,:,:)=S.^(k-1);
   end

   MM=zeros(size(S));
   for k=1:poczp
       M=zeros(size(S));
       for m=1:rad+1
           M=M+squeeze(Sp(m,:,:))*P(k,rad+2-m);
       end
       MM=MM+M.*exp(-Z(k)*S);
   end

  if varargin{5}==1
       figure
       [Ci,Hi]=contour(beta,omega,imag(MM),[0 0],'r--');
       hold on;
       [Cr,Hr]=contour(beta,omega,real(MM),[0 0],'b-');
   else
       Ci=contourc(beta,omega,imag(MM),[0 0]);
       Cr=contourc(beta,omega,real(MM),[0 0]);
  end
   
   
   %detecting the intersection points
   Crr=Cr(1,:)+Cr(2,:)*j;
   for k=1:rad+1;
       Crp(k,:,:)=Crr.^(k-1);
   end

   CMr=zeros(size(Crr));
   for k=1:poczp
       CM=zeros(size(Crr));
       for m=1:rad+1
           CM=CM+squeeze(Crp(m,:,:)).'*P(k,rad+2-m);
       end
       CMr=CMr+CM.*exp(-Z(k)*Crr);
   end

   MI=imag(CMr);
    
   %Extraction of the function in the handle form
   F='0+';
   N='0+';
   for k=1:poczp
       f=num2str(P(k,rad+1));
       n='0';
       for m=1:rad
           f=strcat(f,'+',num2str(P(k,m)),'*','s.^',num2str(rad-m+1));
           n=strcat(n,'+',num2str(P(k,m)*(rad-m+1)),'*','s.^',num2str(rad-m));
       end
       F=strcat(F,'(',f,').*exp(-s*',num2str(Z(k)),')+');
       N=strcat(N,'(',f,').*exp(-s*',num2str(Z(k)),').*',num2str(-Z(k)),'+','(',n,').*exp(-s*',num2str(Z(k)),')+');
   end
   F=strcat(F,'0');
   FF=F;
   FF=strrep(FF,'+-','-');
   Funfin=str2num(strcat('@(s)(',FF,')'));
   N=strcat(N,'0');
   %YResult.dF=strcat('@(s)(',N,')');
   N=strcat('@(s)(',F,')./(',N,')');
   N=str2num(N);
   %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
else
    %function handle based formulation of the function

if nargin==4
    varargin{4}=-1;
end 
    
if nargin==3
    varargin{3}=-1;
    varargin{4}=-1;
end    

if nargin==2
    varargin{2}=-1;
    varargin{3}=-1;
    varargin{4}=-1;
end

    F=varargin{1};
    LengthF=length(char(F));
    if varargin{3}==-1;
        if LengthF>500
            ds=(Reg(2)-Reg(1))*(Reg(4)-Reg(3))/500;
        else
            ds=(Reg(2)-Reg(1))*(Reg(4)-Reg(3))/1000;
        end
    else
        ds=varargin{3};
    end
    
%     %Check the size of grid wrt the region size
%     if ds<(Reg(2)-Reg(1))*(Reg(4)-Reg(3))/3e5&LengthF<500|ds<(Reg(2)-Reg(1))*(Reg(4)-Reg(3))/2e4&LengthF>500;
%         YResult.flag=-1;
%         display('Failure -1: too small grid size for the given region')
%         return
%     end
    
    if varargin{2}==-1;
        e=ds/1e6;
    else
        e=varargin{2};
    end
    
    %slight extension of the region
    bmin=Reg(1)-3*ds;
    bmax=Reg(2)+3*ds;
    wmin=Reg(3)-3*ds;
    wmax=Reg(4)+3*ds;
    
   %mapping the zerolevel curves of the function
   beta=bmin:ds:bmax;
   omega=wmin:ds:wmax;
   [B W]=meshgrid(beta,omega);
   S=B+j*W;
   if varargin{4}==1
       if nargin>=6
           hold off;
       else
           figure
       end
       [Ci,Hi]=contour(beta,omega,imag(F(S)),[0 0],'r--');
       hold on;
       [Cr,Hr]=contour(beta,omega,real(F(S)),[0 0],'b-');
   else
       Ci=contourc(beta,omega,imag(F(S)),[0 0]);
       Cr=contourc(beta,omega,real(F(S)),[0 0]);
   end


   %detecting the intersection points
   Crr=Cr(1,:)+Cr(2,:)*j;
   MI=imag(F(Crr));
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
end

%detecting the intersection points within the given region
Ind=find(MI(2:end).*MI(1:end-1)<=0&Cr(1,2:end)>=bmin&Cr(1,2:end)<=bmax&Cr(2,2:end)>=wmin&Cr(2,2:end)<=wmax&abs(Cr(1,2:end)-Cr(1,1:end-1))<2*ds&abs(Cr(2,2:end)-Cr(2,1:end-1))<2*ds);

%removing multiple root detection
Ind2=zeros(1,length(Cr));
Ind2(Ind)=1;
Ind2=find(Ind2(1:end-1)==1&(Ind2(1:end-1)+Ind2(2:end)~=2));
%plot(Cr(1,Ind2),Cr(2,Ind2),'ko','MarkerSize',3)
%axis(Reg)
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

%Numerical method for increasing accuracy
if isnumeric(varargin{1})
    R=Cr(1,Ind2)+j*Cr(2,Ind2);
    Rback=R;
    Newton_run=1;
    countN=0;
    reduce_ds3=0;
    while Newton_run
        Rk1=R;
        R=R-N(R);
        Newton_run=max(abs(R-Rk1))>0.1*e;
        countN=countN+1;
        if countN>100
            reduce_ds3=1;
            break
        end
        %plot(real(R),imag(R),'o')
    end
    
    kkr=0;
    Rr=R;
    Rcheck=[];
    for k=1:length(R)
        if real(R(k))>=Reg(1)&real(R(k))<=Reg(2)&imag(R(k))>=Reg(3)&imag(R(k))<=Reg(4)
            kkr=kkr+1;
            Rcheck(kkr)=R(k);
        end
    end
    R=Rcheck;
    
    YResult.zeros=sort(R);
    YResult.flag=1;
    YResult.accuracy=10^ceil(log10(e));
    if varargin{5}==1
        plot(real(R),imag(R),'ko','MarkerFaceColor',[0 0 0],'MarkerSize',2)
        title('Quasi-polynomial zeros')
        xlabel('Re(s)')
        ylabel('Im(s)')
    end
    
    
    %Checking the distance from the first approximation of the roots
    %should be less than ds
    reduce_ds1=1;
    k=1;
    brk=1;
    if ~isempty(Rback)
    while min(abs(Rback(k)-Rr))<2*ds&brk
        [Vi I]=min(abs(Rback(k)-Rr));
        Rr(I)=Inf;
        k=k+1;
        if k>length(R)
            k=k-1;
            brk=0;
        end
    end
    if k==length(R)
        reduce_ds1=0;
    end
    end
    
    %checking by argument principle
    Ir=argp(Funfin,Reg,ds/10,e);
    Ir2=argp(Funfin,Reg+[ds/10 -ds/10 ds/10 -ds/10],ds/10,e);%a bit smaller region
    reduce_ds2=1;
    if Ir==length(R)|Ir2==length(R)
        reduce_ds2=0;
    end    
    Yr=[];
    if reduce_ds1+reduce_ds2+reduce_ds3;
        if varargin{4}~=-1
            if varargin{5}==1  
                      display('Failure -2: too large grid size')
            end
            YResult.flag=-1; %opr -2
        else
            if varargin{5}==1
                      display(['Grid s ize reduced from ', num2str(ds), ' to ',num2str(ds/4)])
            end
            ds=ds/3;
            Yr=QPmR(Reg,Funfin,varargin{3},ds,varargin{5},1);
            if Yr.flag==-2
                YResult.flag=0;
                if varargin{5}==1 
                      display(['Failure 0: too large region'])
                end
            else
                YResult.accuracy=Yr.accuracy;
                YResult.zeros=Yr.zeros;
                YResult.flag=Yr.flag;
            end
        end
    end
    
   %including asymptotic curves
   if rad~=0
      grph=varargin{5};
      Y=QPSD(P,Z,grph);
      if ~isempty(Y);
        YResult.asympt=Y;
        mi=Y(:,1);
        c=Y(:,2);
        omegac=eps:0.01*ds:wmax;
        for k=1:length(mi)
             if mi(k)>0
                betac=c(k)-mi(k)*log(omegac);
                if grph==1
                    subplot(2,1,2)
                    hold on
                    plot(betac,omegac,'r-','LineWidth',0.5);
                    hold on
                    plot(betac,-omegac,'r-','LineWidth',0.5);
                end
             end
        end
      else
          if grph==1
              subplot(2,1,2)
              hold on
          end
              
      end    
   else
       YResult.asympt=[];
       %Difference equation only
       if isnumeric(varargin{1})&varargin{5}==1&nargin<7
        figure
        plot(real(YResult.zeros),imag(YResult.zeros),'ko','MarkerFaceColor',[0 0 0],'MarkerSize',2)
        title('Spectrum distribution')
        xlabel('\Re(s)')
        ylabel('\Im(s)')
        grid
        if isempty(YResult.zeros)
            axis(Reg)
        else
            Regsp=[min(real(YResult.zeros)) max(real(YResult.zeros)) min(imag(YResult.zeros)) max(imag(YResult.zeros))];
            Regsp=[max([Reg(1),Regsp(1)-0.1*(Regsp(2)-Regsp(1))]), min([Reg(2),Regsp(2)+0.1*(Regsp(2)-Regsp(1))]), max([Reg(3),Regsp(3)-0.1*(Regsp(4)-Regsp(3))]), min([Reg(4),Regsp(4)+0.1*(Regsp(4)-Regsp(3))])];
            axis(Regsp)
        end
       %computation of safe upper bound
       PD=abs(P);
       IZ0=find(Z==0);
       PD=PD./PD(IZ0);
       PD(IZ0)=0;
       F='0+';
       for k=1:poczp
          f=num2str(PD(k,1));
          F=strcat(F,'(',f,').*exp(-s*',num2str(Z(k)),')+');
       end
       F=strcat(F,'0');
       F=strcat('@(s)(',F,')-1');
       F=str2num(F);
       CD=fzero(F,0);
       YResult.upbound=CD;
       if varargin{5}==1
           hold on
           plot([CD CD],[wmin wmax],'b')
       end
       end
   end
   if varargin{5}==1&rad~=0
        plot(real(YResult.zeros),imag(YResult.zeros),'ko','MarkerFaceColor',[0 0 0],'MarkerSize',2)
        title('Spectrum distribution')
        xlabel('\Re(s)')
        ylabel('\Im(s)')
        grid
        %if isempty(YResult.zeros)
            axis(Reg)
        %else
        %    Regsp=[min(real(YResult.zeros)) max(real(YResult.zeros)) min(imag(YResult.zeros)) max(imag(YResult.zeros))];
        %    Regsp=[max([Reg(1),Regsp(1)-0.1*(Regsp(2)-Regsp(1))])-ds, min([Reg(2),Regsp(2)+0.1*(Regsp(2)-Regsp(1))])+ds, max([Reg(3),Regsp(3)-0.1*(Regsp(4)-Regsp(3))])-ds, min([Reg(4),Regsp(4)+0.1*(Regsp(4)-Regsp(3))])+ds];
        %    axis(Regsp)
        %end
        sp2=gca;
        HH=get(sp2,'Position');
        set(sp2,'Position',[HH(1) HH(2)-0.05*HH(4) HH(3) 1.5*HH(4)])
   end
   YResult.function=Funfin;
   if isempty(Yr)
       YResult.grid=ds;
   else
       YResult.grid=Yr.grid;
   end
       
   if sum((P(:,1)'~=0))>1&rad~=0
       PD=P(:,1);
       RD=QPmR(Reg,PD,Z,varargin{3},varargin{4},varargin{5},1);
       YResult.DEzeros=RD.zeros;
       YResult.DEflag=RD.flag;
       YResult.DEaccuracy=RD.accuracy;
       if varargin{5}==1
             title('Associated difference equation spectrum')
             xlabel('\Re(s)')
             ylabel('\Im(s)')
             figure(gcf-1)
             plot(real(RD.zeros),imag(RD.zeros),'b+')
       end
       %computation of safe upper bound
       PD=abs(PD);
       IZ0=find(Z==0);

       
       PD=PD./PD(IZ0);
       PD(IZ0)=0;
       F='0+';
   for k=1:poczp
       f=num2str(PD(k,1));
       F=strcat(F,'(',f,').*exp(-s*',num2str(Z(k)),')+');
   end
   F=strcat(F,'0');
   F=strcat('@(s)(',F,')-1');
   F=str2num(F);
   CD=fzero(F,0);
   YResult.DEupbound=CD;
   YResult.DEfunction=RD.function; 
   YResult.DEgrid=RD.grid;
   if varargin{5}==1
       plot([CD CD],[wmin wmax],'b')
   end
   end
   
    %reorder the outputs
    if nargin<=6
        if nargout<=1
            if YResult.flag==1
                YResult=YResult.zeros';
            else
                YResult=NaN;
            end              
        elseif nargout==2
            YResultz=YResult;
            if YResult.flag==1
                YResult=YResultz.zeros';
             else
                YResult=NaN;
            end    
            varargout={YResultz};
        else
            display('Too many outputs')
            return
        end
    end
   
else
    %numerical evaluation of the derivative
    R=Cr(1,Ind2)+j*Cr(2,Ind2);
    Rback=R;
    Newton_run=1;
    de=e/100;
    countN=0;
    reduce_ds3=0;
    while Newton_run
        Rk1=R;
        dF=((F(R)-F(R+de))./de-(F(R)-F(R-de))./de+(F(R)-F(R+j*de))./j/de-(F(R)-F(R-j*de))./j/de)/4;
        R=R+F(R)./dF;
        Newton_run=max(abs(R-Rk1))>0.1*e;

        countN=countN+1;
        if countN>100
            reduce_ds3=1;
            break
        end
    end
    
    %Removing the roots outside the region Reg
    
    kkr=0;
    Rcheck=[];
    Rr=R;
    for k=1:length(R)
        if real(R(k))>=Reg(1)-e&real(R(k))<=Reg(2)+e&imag(R(k))>=Reg(3)-e&imag(R(k))<=Reg(4)+e
            kkr=kkr+1;
            Rcheck(kkr)=R(k);
        end
    end
    R=Rcheck;
    
    YResult.zeros=sort(R);
    YResult.flag=1;
    YResult.accuracy=10^ceil(log10(e));
    YResult.grid=ds;

    %Checking the distance from the first approximation of the roots
    %should be less than 2ds
    reduce_ds1=1;
    k=1;
    brk=1;
    if ~isempty(Rback)&~isempty(R)
    while min(abs(Rback(k)-Rr))<2*ds&brk
        [Vi I]=min(abs(Rback(k)-Rr));
        Rr(I)=Inf;
        k=k+1;
        if k>length(R)
            k=k-1;
            brk=0;
        end
    end
    if k==length(R)
        reduce_ds1=0;
    end
    else
        reduce_ds1=0;
    end
    
    %checking by argument principle
    Ir=argp(F,Reg,ds/10,e);
    Ir2=argp(F,Reg+[ds/10 -ds/10 ds/10 -ds/10],ds/10,e);%a bit smaller region
    reduce_ds2=1;
    if Ir==length(R)|Ir2==length(R)
        reduce_ds2=0;
    end    
    
    %Computation with half grid
    if ~(reduce_ds1+reduce_ds2+reduce_ds3)&(varargin{3}==-1|nargin>5)
        if isempty(R)
            Reghr=Reg;
            dsr=ds/3;
        elseif length(R)==1|abs(sum(R)/length(R)-R(1))<ds
            Reghr=[max(real(R(1))-5*ds, Reg(1)) min(real(R(1))+5*ds, Reg(2)) max(imag(R(1))-5*ds, Reg(3)) min(imag(R(1))+5*ds, Reg(4))]; %mozna by  neml byt ymetricky kdyz je u kraje
            dsr=ds/100;                 
        else
            Reghr=[min(real(R))-e max(real(R))+e min(imag(R))-e max(imag(R))+e];
            dsr=(Reg(2)-Reg(1)+Reg(4)-Reg(3))/ds;
            dsr=(Reghr(2)-Reghr(1)+Reghr(4)-Reghr(3))/dsr/2;
            
        end
        Rh=QPmR(Reghr,F,e,dsr);
        Rhr=sort(Rh+j*eps);
        Rhro=sort(R+j*eps);
        if ~(isempty(R)&isempty(Rhr))&length(R)==length(Rhr)
            condEmpty=sum((find(abs(abs(Rhr)-abs(Rhro'))>2*e)));
            if condEmpty>0 
               reduce_ds1=1; 
            end
        end
        
        if sum(isnan(Rhr))|length(R)~=length(Rhr)
            reduce_ds1=1; 
        end
        R=[];
        R=Rh;
    end
    
       
    
    brk=1;
    if nargin>=6
        if varargin{5}==2
            brk=0;
            if reduce_ds1+reduce_ds2
                if varargin{4}==1 
                     display(['Failure 0: too large region'])
                end
                YResult.flag=0;
            end
        end
    end
    
    if isempty(R)
        reduce_ds1=0;
    end
            
    
    if (reduce_ds1+reduce_ds2+reduce_ds3)*brk;
        if varargin{3}~=-1&length(varargin)+1==5
            if varargin{4}==1 
                display('Failure -1: too large grid size')
            end
            YResult.flag=-1; %opr -2
        else
        if nargin>=6
            if varargin{4}==1 
                display(['Grid size reduced from ', num2str(ds), ' to ',num2str(ds/4)])
            end
            ds=ds/3;
            if nargin<7
                varargin{6}=0;
            end
            Y=QPmR(Reg,F,varargin{2},ds,varargin{4},varargin{5}+1,varargin{6});
            if Y.flag==-1
                YResult.flag=0;
                if varargin{4}==1
                    display(['Failure 0: too large region'])
                end
            else
                YResult.accuracy=Y.accuracy;
                YResult.zeros=Y.zeros;
                YResult.flag=Y.flag;
                YResult.grid=Y.grid;
            end
        else
            if varargin{4}==1
                display(['Grid size reduced from ', num2str(ds), ' to ',num2str(ds/4)])
            end
            ds=ds/3;
            if nargin<7
                varargin{6}=0;
            end
                
            Y=QPmR(Reg,F,varargin{2},ds,varargin{4},1,varargin{6});
            if Y.flag==-2
                YResult.flag=0;
                if varargin{4}==1
                    display(['Failure 0: too large region'])
                end
            else
                YResult.accuracy=Y.accuracy;
                YResult.zeros=Y.zeros;
                YResult.flag=Y.flag;
                YResult.grid=Y.grid;
            end
        end
        end
    end
    
    %decreasing the region by recursion
    if YResult.flag==0&brk==0
        if varargin{6}==0
            if varargin{4}==1
                clf
            end
            ffg=0;
            if varargin{4}==1 
                display('Recursion: Level 1')
                display(['Grid size reduced from ', num2str(YResult.grid), ' to ', num2str(YResult.grid/4)])
                display('Subregion 1')
                ffg=-2;
            end
            ds=YResult.grid/4;
            Y1=QPmR([Reg(1) Reg(1)+0.5*(Reg(2)-Reg(1)) Reg(3) Reg(3)+0.5*(Reg(4)-Reg(3))],F,varargin{2},ds,ffg,2,1);
            if varargin{4}==1
                display('Subregion 2')
            end
            Y2=QPmR([Reg(1) Reg(1)+0.5*(Reg(2)-Reg(1)) Reg(3)+0.5*(Reg(4)-Reg(3))+e Reg(4)],F,varargin{2},ds,ffg,2,1);
            if varargin{4}==1
                display('Subregion 3')
            end
            Y3=QPmR([Reg(1)+0.5*(Reg(2)-Reg(1))+e Reg(2) Reg(3) Reg(3)+0.5*(Reg(4)-Reg(3))],F,varargin{2},ds,ffg,2,1);
            if varargin{4}==1
                display('Subregion 4')
            end
            Y4=QPmR([Reg(1)+0.5*(Reg(2)-Reg(1))+e Reg(2) Reg(3)+0.5*(Reg(4)-Reg(3))+e Reg(4)],F,varargin{2},ds,ffg,2,1);           
            YResult.zeros=sort([Y1.zeros, Y2.zeros, Y3.zeros, Y4.zeros]);
            YResult.flag=Y1.flag*Y2.flag*Y3.flag*Y4.flag;
            YResult.grid=min([Y1.grid, Y2.grid, Y3.grid, Y4.grid]);
            YResult.accuracy=max([Y1.accuracy, Y2.accuracy, Y3.accuracy, Y4.accuracy]);
        elseif varargin{6}<2
            varargin{6}=varargin{6}+1;
            if varargin{4}==-2
                display('Recursion: Level 2')
                display(['Grid size reduced from ', num2str(YResult.grid), ' to ', num2str(YResult.grid/4)])
                display('Sub-subregion 1')
            end
            ds=YResult.grid/4;
            Y1=QPmR([Reg(1) Reg(1)+0.5*(Reg(2)-Reg(1)) Reg(3) Reg(3)+0.5*(Reg(4)-Reg(3))],F,varargin{2},ds,0,2,varargin{6});
            if varargin{4}==-2
                display('Sub-subregion 2')
            end
            Y2=QPmR([Reg(1) Reg(1)+0.5*(Reg(2)-Reg(1)) Reg(3)+0.5*(Reg(4)-Reg(3))+e Reg(4)],F,varargin{2},ds,0,2,varargin{6});
            if varargin{4}==-2
                display('Sub-subregion 3')
            end
            Y3=QPmR([Reg(1)+0.5*(Reg(2)-Reg(1))+e Reg(2) Reg(3) Reg(3)+0.5*(Reg(4)-Reg(3))],F,varargin{2},ds,0,2,varargin{6});
            if varargin{4}==-2
                display('Sub-subregion 4')
            end
            Y4=QPmR([Reg(1)+0.5*(Reg(2)-Reg(1))+e Reg(2) Reg(3)+0.5*(Reg(4)-Reg(3))+e Reg(4)],F,varargin{2},ds,0,2,varargin{6});
            YResult.zeros=sort([Y1.zeros, Y2.zeros, Y3.zeros, Y4.zeros]);
            YResult.flag=Y1.flag*Y2.flag*Y3.flag*Y4.flag;
            YResult.grid=min([Y1.grid, Y2.grid, Y3.grid, Y4.grid]);
            YResult.accuracy=max([Y1.accuracy, Y2.accuracy, Y3.accuracy, Y4.accuracy]);            
        end
    end
    
    %figure
    if varargin{4}==1
        plot(real(YResult.zeros),imag(YResult.zeros),'ko','MarkerFaceColor',[0 0 0],'MarkerSize',2)
        axis(Reg)
        title('Quasi-polynomial spectrum')
        xlabel('\Re(s)')
        ylabel('\Im(s)')
        axis(Reg)
        grid
    end
    
    %reorder the outputs
    if nargin<=5
        if nargout<=1
            if YResult.flag==1
                YResult=YResult.zeros.';
            else
                YResult=NaN;
            end              
        elseif nargout==2
            YResultz=YResult;
            if YResult.flag==1
                YResult=YResultz.zeros.';
             else
                YResult=NaN;
            end    
            varargout={YResultz};
        else
            display('Too many outputs')
            return
        end
    end
    
end


        


%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%Subfunction for constructing distribution diagram
function Y=QPSD(P,Z,grph)
[Del I]=sort(Z);
Pol=P(I,:);

max_order=size(Pol);
max_order=max_order(2);
N_Del=length(Del);

for k=1:N_Del
    p=1;
    while Pol(k,p)==0
        p=p+1;
    end
    E_pol(1,k)=max_order-p;
    P_pol(1,k)=Pol(k,p);
end  

for k=1:N_Del
    E_pr(N_Del-k+1)=E_pol(k);
    P_pr(N_Del-k+1)=P_pol(k);
end

Del_pr=zeros(1,N_Del);
for k=2:N_Del
    Del_pr(k)=Del(N_Del)-Del(N_Del-k+1);
end

if grph==1
    h1=figure;
    HH=get(h1,'Position');
    set(h1,'Position',[HH(1) 0.3*HH(2) HH(3) HH(4)+0.7*HH(2)]);
    subplot(2,1,1)
    plot(Del_pr,E_pr,'o','Markersize',3)
    axis([Del_pr(1) Del_pr(end) 0 max(E_pr)+1])
    grid
    sp1=gca;
    HH=get(sp1,'Position');
    set(sp1,'Position',[HH(1) HH(2)+0.45*HH(4) HH(3) 0.6*HH(4)])
end

p=1;
n_d=N_Del;
Del_r=Del_pr(p);
E_r=E_pr(1);
P_P=P_pr(1);
E_P=E_pr(1);
r_mi=1;
Del_P(1)=Del_r;

while p~=N_Del
    for k=p+1:n_d
        mi_semi(k-p)=(E_pol(N_Del-k+1)-E_r)/(Del_pr(k)-Del_r);
    end
        [mi(r_mi),b] = max(mi_semi);
        p=b+p;
        E_r=E_pr(p);
        E_P(r_mi+1)=E_r;
        Del_r=Del_pr(p);
        Del_P(r_mi+1)=Del_r;
        P_P(r_mi+1)=P_pr(p);
        r_mi=r_mi+1;
        mi_semi=0;
end

if grph==1
    hold on
    plot(Del_P,E_P,'k-')

    title('Spectrum distribution diagram')    
    xlabel('Delay','FontSize',10)
    ylabel('Polynom degree','FontSize',10)
end

for k=1:r_mi
    P_Pp(k)=P_P(r_mi-k+1);
end

if sum(mi)~=0
k=1;
kp=1;
while k<r_mi
    clear Koef
    
    if k<r_mi-1
        podm=mi(k)~=mi(k+1);
    elseif r_mi==2
        podm=1;
    else
        podm=mi(k)~=mi(k-1);
    end
    
    if podm
        nP=E_P(k+1)-E_P(k);
        Koef(1)=P_P(k+1);
        Koef(nP+1)=P_P(k);
                
        w_P(1:nP)=roots(Koef);
        c(kp)=mi(k)*log(abs(w_P(1)));
        mi_bud(kp)=mi(k);
        k=k+1;
        kp=kp+1;
    else
        kk=k;
        i=k;
        opakuj=(mi(i)==mi(i+1));
        while opakuj
            kk=kk+1;
            if i==r_mi-1;
                opakuj=0;
            else
                opakuj=(mi(i)==mi(i+1));
                i=i+1;
            end
        end
        nP=0;
        Koef(1)=P_P(kk);
        for i=1:(kk-k)
            nP=nP+E_P(kk-i+1)-E_P(kk-i);
            Koef(nP+1)=P_P(kk-i);
        end 
        
        w_P(1:nP)=roots(Koef);
        
        c(kp)=mi(k)*log(abs(w_P(1)));
        mi_bud(kp)=mi(k);
        kp=kp+1;
        
        for i=2:nP
            if abs(w_P(i))~=abs(w_P(i-1))
                c(kp)=mi(k)*log(abs(w_P(i)));
                mi_bud(kp)=mi(k);
                kp=kp+1;
            end
        end
              
        k=kk;
    end   
end
mi=mi_bud;

Y=mi';
Y(:,2)=c';
else
    Y=[];
end


%subfunction for checking the number of roots
function Ir=argp(F,Reg,ds,e)
de=e/100;
ds=ds;

Reg=Reg+[-ds ds -ds ds];
Rx=linspace(Reg(1),Reg(2),floor((Reg(2)-Reg(1))/ds)+1);
dsx=(Reg(2)-Reg(1))/(length(Rx)-1);
Ry=linspace(Reg(3),Reg(4),floor((Reg(4)-Reg(3))/ds)+1);
dsy=(Reg(4)-Reg(3))/(length(Ry)-1);
R=[Rx+j*Reg(3), Reg(2)+j*Ry, fliplr(Rx)+j*Reg(4), Reg(1)+fliplr(Ry)*j];
dS=[ones(1,length(Rx))*dsx, j*ones(1,length(Ry))*dsy, -ones(1,length(Rx))*dsx, -j*ones(1,length(Ry))*dsy];

dF=((F(R)-F(R+de))./de-(F(R)-F(R-de))./de+(F(R)-F(R+j*de))./j/de-(F(R)-F(R-j*de))./j/de)/4;
Ir=round(abs(real(sum(dF./F(R).*dS)/(2*pi*j))));



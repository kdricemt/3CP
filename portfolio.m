%% Portfolio Problem

% Problem Value at Risk Minimation Problem
%  min J(x)=t
%  s.t. Prob{Sum(r_j x_j) < t} <= epsilon
%       Sum(x_j) = 1, x_j > 0 
clear all; close all;

d = 20;  % Number of assets

scenario  = 1;  % Conduct Scenario Appoach
bernstein = 0;  % Conduct Bernstein Approx

%% 1. Nonlinear Optimization with Scenario Approach
% Sample N samples and solve
% %  min J(x)=t
%  s.t. Prob{Sum(r_j x_j) < t} <= epsilon
%       Sum(x_j) = 1, x_j > 0 

if scenario
    eps_array  = 0.01:0.005:0.05;

    ncase      = length(eps_array);
    var_array  = zeros(1,ncase);
    vr_array   = zeros(1,ncase);

    rng(1);

    for ii = 1:ncase
        epsilon = eps_array(ii);
        beta    = 1 -epsilon;

        % Number of required samples
        eps_N = 0.01;
        N = ceil(2/(1-beta)*log(1/eps_N) + 2*d + 2*d/(1-beta)*log(2/(1-beta)));

        % create random variables
        % r = (ones(d,N) + 0.06*jvec) + 0.04*(jvec.*xi);  % d X N  non-affine
        r = weight(d,N);  % d X N affine

        % Test
        x0 = 1/d * ones(d,1);
        y0 = [x0; 2];

        % Linear Programming
        fobj = [zeros(1,d),-1];
        A   = [-r', ones(N,1)];  %N x (d+1) t - rx <= 0  
        b   = zeros(N,1);
        Aeq = [ones(1,d),0];
        beq = 1;
        lb  = zeros(d+1,1);
        ub  = [ones(d,1);2];

        % Solutions
        ysol = linprog(fobj,A,b,Aeq,beq,lb,ub);
        xsol = ysol(1:end-1);
        varsol = ysol(end);

        % calculate violation rate
        Nv = 10^6;
        vr = Vrate(xsol,varsol,Nv);

        % Print Results
    %     disp(['eps:',num2str(epsilon)]);
    %     disp(['    VaR:',num2str(tsol)]);
    %     disp(['    Violation Rate:',num2str(vr)]);

        var_array(ii) = varsol;
        vr_array(ii) = vr;
    end

    figure(1);
    hold on; grid on;
    title('Sampling Method');
    xlabel('eps')
    ylabel('VaR')
    plot(eps_array,var_array,'LineWidth',4);
    set(gca,'FontSize',14);

    figure(2);
    hold on; grid on;
    title('Sampling Method');
    xlabel('eps')
    ylabel('Violation Rate')
    plot(eps_array,vr_array,'LineWidth',4);
    set(gca,'FontSize',14);

end
%% Bernstein Approximation

% solve by NLP solver (fmincon)
if bernstein
    
    % Init value
    x0 = 1/d * ones(d,1);  % x, VaR, t(bernstein approx param)
    y0 = [x0; 2; 2];
    
    % Linear Programming
    A   = [];  %N x (d+1) t - rx <= 0  
    b   = [];
    Aeq = [ones(1,d),0, 0];
    beq = 1;
    lb  = zeros(d+2,1);
    ub  = [ones(d,1);2;10];
    
    alpha = 0.01;
    
    eps = 1e-6;
    delta = 0.0025;
    xirange = norminv([eps/2,1-eps/2]);
    xivec  = xirange(1):delta:xirange(2);  % a_k, k=1,2,..n
    Pvec   = normcdf(xivec);  % k = 1,...n
    nonlconf = @(y)nonlcol_bernstein(y,alpha,xivec,Pvec);
    
    ysol = fmincon(@fobj_bernstein,y0,A,b,Aeq,beq,lb,ub,nonlconf);
    xsol = ysol(1:end-2);
    varsol = ysol(end-1);
    tsol = ysol(end);

    % calculate violation rate
    Nv = 10^6;
    vr = Vrate(xsol,varsol,Nv);
    
    [c,~] = nonlconf(ysol);
    
    disp(['C (<0):',num2str(c)]);
    
    disp('Bernstein');
    disp(['  VaR:',num2str(varsol)]);
    disp(['  Viloation Rate:',num2str(vr)]);
end


%% Supplementary Function

% Objective function
function r = weight(d,N)
    eta = random('Normal',0,1,d,N);   % d x N
    j = repmat((1:d)',1, N);
    
    r = (ones(d,N) + 0.06*j) + 0.04*(j.*eta);  % d X N, non-affine
end

% calculate vilation rate
function vr = Vrate(x,t,N)
   d = length(x);
   r = weight(d,N);
   
   gvec = t - x'*r;  % 1 x N
   gv = find(gvec>0);
   vr = length(gv)/N;
end

function out = fobj_bernstein(y)
    d = length(y)-2;
    out = [zeros(1,d),-1,0] * y;
end

function out = integ_bernstein(xivec, Pvec,t, gi)
    gamma = log(trapz(Pvec, exp(xivec * gi)));
    out = t * gamma;
end

function [c,ceq] = nonlcol_bernstein(y,alpha,xivec,Pvec)
   d = length(y)-2;
   
   x   = y(1:end-2);
   var = y(end-1);
   t   = y(end);
   
   g0 = var - sum(x);  
   
   itg = 0;
   
   for di = 1:d
      gi = - 0.04*di*x(di);
      itg = itg + integ_bernstein(xivec + 1.5, Pvec,t, gi);
   end
   
   c = g0 + itg - t*log(alpha);
   ceq = [];
end
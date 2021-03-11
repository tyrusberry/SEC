clear;close all;
addpath('SEC');

    %%%%%%%%%%%%%%%%%%%%%%%%% GENERATE DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    i=5;    %%% Which of the examples below to run
            %%% Note: i=2 is a long run due to using full matrices in
            %%%         the diffusion maps algorithm
    
    examples = {'circle','flattorus','torus','doubletorus','L63','sphere','mobius'};
    Ns = [101;10000;5000;12000;4000;10000;1000];    %%% number of data points
    epsilons = [.05;.05;0.2;.1;2;.1;.1];            %%% bandwidth parameter for diffusion maps
    
    n0 = 100;           %%% number of eigenfunctions for computing c_{ijk}
    n1 = 20;            %%% number of eigenfunctions in the SEC frame (n1^2 frame elements)
    rng(3);
    [x,intrinsic] = GenerateDataSet(Ns(i),examples{i});

    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SEC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Construct the 0-Laplacian using Diffusion Maps

    [u,l,D] = Del0(x,n0,epsilons(i));  


    %%% Construct the 1-Laplacian using the SEC (anti-symmetric frame)

    [U,L,D1,G,H,cijk]=Del1AS(u,l,D,n1);
    
    
    %%% Alternative construction using the b^idb^j frame (non-anti-symmetric)
    %[U,L,D1,G,cijk]=Del1(u,l,D,n1); 
    %H=G;
    
    
    L(1:8)    %%% number of zero eigenvalues is the first Betti number
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%% Draw eigen-vectorfields %%%%

    xhat = x*D*u(:,1:n1);          %%% Fourier transform of the embedding coordinates
    ihat = intrinsic*D*u(:,1:n1);  %%% Fourier transform of the intrinsic coordinates

    for j = 1:4
        
        %%% Convert j-th eigenfield U(:,j) from frame representation to 
        %%% operator representation Umatrix
        Umatrix = reshape(H'*U(:,j),n1,n1);             
        
        %%% Apply the vector field to the desired coordinates by
        %%% multiplying with the Fourier tranform of the coordinates 
        datavectorfield = xhat*Umatrix'*u(:,1:n1)'; 
        intrinsicvectorfield = ihat*Umatrix'*u(:,1:n1)';
    
        figure(1);
        subplot(2,2,j); 
        plot3(x(1,:),x(2,:),x(3,:),'.b','markersize',10);hold on;
        quiver3(x(1,:),x(2,:),x(3,:),datavectorfield(1,:),datavectorfield(2,:),datavectorfield(3,:),1,'r');
        
        figure(2);
        subplot(2,2,j); 
        quiver(intrinsic(1,:),intrinsic(2,:),intrinsicvectorfield(1,:),intrinsicvectorfield(2,:),1,'r');
        
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% PLOTS FROM PAPER %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%% For the circle and flat torus examples we can compare the spectrum
    %%% of the 1-Laplacian from the SEC to the analytic spectrum
    
    if (i==1)
        circleEigs = repmat((1:30).^2,2,1);
        circleEigs = [0;circleEigs(:)];

        figure(3);
        plot(circleEigs,'linewidth',4,'color',[.7 .7 .7]);
        hold on;
        plot(L,'r--','linewidth',2);
        l=legend('Truth','SEC','location','northwest');set(l,'fontsize',22);
        set(gca,'fontsize',20);
        xlabel('n','fontsize',24);
        ylabel('\lambda_n, eigenvalues of \Delta_1','fontsize',24);
        xlim([1 41]);ylim([0 500]);
        
        for j=1:4
            skip=4;
            Umatrix = reshape(H'*U(:,j),n1,n1);              
            intrinsicvectorfield = ihat*Umatrix'*u(:,1:n1)';
            figure(j+4);
            plot(intrinsic(1,1:1:end),intrinsic(2,1:1:end),'.b','markersize',25);hold on;
            quiver(intrinsic(1,1:skip:end),intrinsic(2,1:skip:end),intrinsicvectorfield(1,1:skip:end),intrinsicvectorfield(2,1:skip:end),1,'r','linewidth',2);
            xlim([min(intrinsic(1,:))*1.17 max(intrinsic(1,:))*1.17]);
            ylim([min(intrinsic(2,:))*1.17 max(intrinsic(2,:))*1.17]);
            set(gca,'fontsize',20);
            xlabel('x','fontsize',24);
            ylabel('y','fontsize',24);
        end
    end
    
    if (i==2)
        ll=[];
        %%% Lazy way to generate the true spectrum on the flat torus (only
        %%% the beginning of the sequence is correct - only that is used)
        ll(1)=0;
        for j = 1:20
            ll(end+1:end+4)=j^2;
        end
        for i=1:10
            for j = 1:10
                ll(end+1:end+4)=i^2+j^2;
            end
        end
        ll=sort(ll);
        torusEigs=[]; %%% True 1-Laplacian spectrum on flat torus
        cc=0;
        ss=unique(ll);
        for i=1:60
            %%% multiplicities of the 1-Laplacian eigenvalues are double those of the 0-Laplacian
           numi = sum(ll==ss(i));
           torusEigs(cc+(1:2*numi)) = ss(i);
           cc=cc+2*numi;
        end
        figure(3);
        plot(torusEigs,'linewidth',4,'color',[.7 .7 .7]);
        hold on;
        plot(L,'r--','linewidth',2);
        l=legend('Truth','SEC','location','northwest');set(l,'fontsize',22);
        set(gca,'fontsize',20);
        xlabel('n','fontsize',24);
        ylabel('\lambda_n, eigenvalues of \Delta_1','fontsize',24);
        xlim([1 110]);ylim([0 19]);
        
        for j=1:2
            skip=5;
            Umatrix = reshape(H'*U(:,j),n1,n1);              
            xvectorfield = xhat*Umatrix'*u(:,1:n1)';
            figure(j+3); 
            plot(x(1,1:1:end),x(2,1:1:end),'.b','markersize',20);hold on;
            quiver(x(1,1:skip:end),x(2,1:skip:end),xvectorfield(1,1:skip:end),xvectorfield(2,1:skip:end),8,'r','linewidth',.7);
            xlim([min(x(1,:))*1.17 max(x(1,:))*1.17]);
            ylim([min(x(2,:))*1.17 max(x(2,:))*1.17]);
            set(gca,'fontsize',20);
            xlabel('$\cos(\theta)$','fontsize',24,'interpreter','latex');
            ylabel('$\sin(\theta)$','fontsize',24,'interpreter','latex');
            figure(j+5); skip=500;
            plot(x(3,1:1:end),x(4,1:1:end),'.b','markersize',20);hold on;
            quiver(x(3,1:skip:end),x(4,1:skip:end),xvectorfield(3,1:skip:end),xvectorfield(4,1:skip:end),1,'r','linewidth',1);
            xlim([min(x(3,:))*1.17 max(x(3,:))*1.17]);
            ylim([min(x(4,:))*1.17 max(x(4,:))*1.17]);
            set(gca,'fontsize',20);
            xlabel('$\cos(\phi)$','fontsize',24,'interpreter','latex');
            ylabel('$\sin(\phi)$','fontsize',24,'interpreter','latex');
        end
    end
    
    if (i==3)
        for j = 1:4
            Umatrix = reshape(H'*U(:,j),n1,n1);             
            datavectorfield = xhat*Umatrix'*u(:,1:n1)'; 
            intrinsicvectorfield = ihat*Umatrix'*u(:,1:n1)';
            skip = 4;
            figure(j);
            quiver3(x(1,1:skip:end),x(2,1:skip:end),x(3,1:skip:end),datavectorfield(1,1:skip:end),datavectorfield(2,1:skip:end),datavectorfield(3,1:skip:end),2,'r','linewidth',1);axis equal;
            set(gca,'camerapos',[-21.3469  -27.8146   27.3930]);
            set(gca,'fontsize',20);set(gca,'ztick',[-1 0 1]);zlim([-1.1 1.1]);
        end
    end
    
    if (i==4)
        for j = 1:4
            Umatrix = reshape(H'*U(:,j),n1,n1);             
            datavectorfield = xhat*Umatrix'*u(:,1:n1)'; 
            intrinsicvectorfield = ihat*Umatrix'*u(:,1:n1)';

            skip = 4;
            figure(j+2);
            quiver3(x(1,1:skip:end),x(2,1:skip:end),x(3,1:skip:end),datavectorfield(1,1:skip:end),datavectorfield(2,1:skip:end),datavectorfield(3,1:skip:end),2,'r','linewidth',1);axis equal;
            set(gca,'camerapos',[-0.3631  -36.3634   47.9147]);
            set(gca,'fontsize',20);set(gca,'ztick',[-1 0 1]);zlim([-1.1 1.1]);
        end
    end
    
    if (i==5)
        for j=1:4
            skip=1;
            Umatrix = reshape(H'*U(:,j),n1,n1);              
            ivectorfield = ihat*Umatrix'*u(:,1:n1)';
            figure(j+3); 
            quiver(intrinsic(1,1:skip:end),intrinsic(2,1:skip:end),ivectorfield(1,1:skip:end),ivectorfield(2,1:skip:end),2,'r','linewidth',1.3);
            xlim([min(intrinsic(1,:))*1.1 max(intrinsic(1,:))*1.1]);
            ylim([min(intrinsic(2,:))*.9 max(intrinsic(2,:))*1.1]);
            set(gca,'fontsize',20);
            xlabel('x+y','fontsize',24,'interpreter','latex');
            ylabel('z','fontsize',24,'interpreter','latex');
        end
    end
    
    if (i==6)
        for j = 1:4
            Umatrix = reshape(H'*U(:,j),n1,n1);             
            datavectorfield = xhat*Umatrix'*u(:,1:n1)'; 
            skip = 3;
            figure(j+2);
            quiver3(x(1,1:skip:end),x(2,1:skip:end),x(3,1:skip:end),datavectorfield(1,1:skip:end),datavectorfield(2,1:skip:end),datavectorfield(3,1:skip:end),2,'r','linewidth',1);
            set(gca,'fontsize',20);
            xlim([-1.1 1.1]);ylim([-1.1 1.1]);zlim([-1.1 1.1]);
        end
    end
    
    if (i==7)
        for j = 1:4
            Umatrix = reshape(H'*U(:,j),n1,n1);             
            datavectorfield = xhat*Umatrix'*u(:,1:n1)'; 

            skip = 3;
            figure(j+2);
            quiver3(x(1,1:skip:end),x(2,1:skip:end),x(3,1:skip:end),datavectorfield(1,1:skip:end),datavectorfield(2,1:skip:end),datavectorfield(3,1:skip:end),2,'r','linewidth',1);
            set(gca,'fontsize',20);
            xlim([-1.1 1.6]);
            zlim([-.55 .55]);
            set(gca,'camerapos',[-5.6805  -10.2641    8.4112]);
        end
    end
clear all
close all
clc
%Jemone is 85, Malina 74, Benjamin 79,
directory = "./DataTests/";
room32NumStudents = 16;
room20NumStudents = 20;
room18NumStudents = 22;
room16NumStudents = 15;
room6NumStudents = 19;
female = 1;
genderRoom32 = [1;1;0;1;0;0;0;0;1;0;1;1;1;0;0;1];
genderRoom16 = [0;1;0;1;0;1;0;0;0;1;1;1;1;0;0];
genderRoom6 = [0;1;1;1;0;1;1;1;1;0;1;0;0;1;0;1;0;1;1;1;1];
genderRoom18 = [1;1;0;1;0;0;1;1;1;0;1;1;0;0;1;1;1;1;0;1;0;1];
genderRoom20 = [1;0;0;1;1;0;0;1;1;0;0;0;0;0;1;0;1;0;1;1];



maxFingers = 60;
maxMath = 39;
maxPuzzles = 22;
maxCircles = 10;

run('./DataTests/Room32FingersPost.m') %48 is totals %91 is totals counting hand
run('./DataTests/Room32MathPost1.m')
run('./DataTests/Room32MathPre.m')
run('./DataTests/Room32PaperClip.m')
run('./DataTests/Room32Puzzles.m')
run('./DataTests/Room32SFingersPre.m')
run('./DataTests/Room32SubFingersPre.m')
run('./DataTests/Room32SubFingerPost.m')
run('./DataTests/Room20FingersPost.m') %48 is totals %91 is totals counting hand
run('./DataTests/Room20MathPost1.m')
run('./DataTests/Room20MathPre.m')
run('./DataTests/Room20PaperClip.m')
run('./DataTests/Room20Puzzles.m')
run('./DataTests/Room20FingersPre.m')
run('./DataTests/Room20SubFingersPre.m')
run('./DataTests/Room20SubFingersPost.m')
run('./DataTests/Room18FingersPost.m') %48 is totals %91 is totals counting hand
run('./DataTests/Room18MathPost1.m')
run('./DataTests/Room18MathPre.m')
run('./DataTests/Room18PaperClip.m')
run('./DataTests/Room18Puzzles.m')
run('./DataTests/Room18FingersPre.m')
run('./DataTests/Room18SubFingersPre.m')
run('./DataTests/Room18SubFingersPost.m')
run('./DataTests/Room16FingersPost.m') %48 is totals %91 is totals counting hand
run('./DataTests/Room16MathPost1.m')
run('./DataTests/Room16MathPre.m')
run('./DataTests/Room16PaperClip.m')
run('./DataTests/Room16Puzzles.m')
run('./DataTests/Room16FingersPre.m')
run('./DataTests/Room16SubFingersPre.m')
run('./DataTests/Room16SubFingersPost.m')
run('./DataTests/Room6FingersPost.m') %48 is totals %91 is totals counting hand
run('./DataTests/Room6MathPost1.m')
run('./DataTests/Room6MathPre.m')
run('./DataTests/Room6PaperClip.m')
run('./DataTests/Room6Puzzles.m')
run('./DataTests/Room6FingersPre.m')
run('./DataTests/Room6SubFingersPre.m')
run('./DataTests/Room6SubFingersPost.m')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Finger Perception Analysis: %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%without half points:
room32FingersPre1 = takeouthalfpoints(room32FingersPre,room32NumStudents);
room32FingersPost1 = takeouthalfpoints(room32FingersPost,room32NumStudents);
room20FingersPre1 = takeouthalfpoints(room20FingersPre,room20NumStudents);
room20FingersPost1 = takeouthalfpoints(room20FingersPost,room20NumStudents);
room18FingersPre1 = takeouthalfpoints(room18FingersPre,room18NumStudents);
room18FingersPost1 = takeouthalfpoints(room18FingersPost,room18NumStudents);
room16FingersPre1 = takeouthalfpoints(room16FingersPre,room16NumStudents);
room16FingersPost1 = takeouthalfpoints(room16FingersPost,room16NumStudents);
room6FingersPre1 = takeouthalfpoints(room6FingersPre,room6NumStudents);
room6FingersPost1 = takeouthalfpoints(room6FingersPost,room6NumStudents);


%Plot all rooms 
figure
hold on
plot(room32FingersPre1(:,2),room32FingersPre1(:,48),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room32FingersPost1(:,2),room32FingersPost1(:,48),'.','MarkerSize',20,'MarkerFaceColor','cyan','MarkerEdgeColor','cyan')
plot(room20FingersPre1(:,2),room20FingersPre1(:,48),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room20FingersPost1(:,2),room20FingersPost1(:,48),'.','MarkerSize',20,'MarkerFaceColor','cyan','MarkerEdgeColor','cyan')
plot(room18FingersPre1(:,2),room18FingersPre1(:,48),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room18FingersPost1(:,2),room18FingersPost1(:,48),'.','MarkerSize',20,'MarkerFaceColor','cyan','MarkerEdgeColor','cyan')
plot(room16FingersPre1(:,2),room16FingersPre1(:,48),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','red')
plot(room16FingersPost1(:,2),room16FingersPost1(:,48),'.','MarkerSize',20,'MarkerFaceColor','cyan','MarkerEdgeColor','magenta')
plot(room6FingersPre1(:,2),room6FingersPre1(:,48),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','red')
plot(room6FingersPost1(:,2),room6FingersPost1(:,48),'.','MarkerSize',20,'MarkerFaceColor','cyan','MarkerEdgeColor','magenta')
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot all rooms, pre-test and post test: fingers % 60 is max pts
figure
subplot(2,1,1)
hold on
% computers
plot(room16FingersPre(:,2),room16FingersPre(:,48),'o','MarkerSize',10,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room6FingersPre(:,2),room6FingersPre(:,48),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
%hapcaps
plot(room32FingersPre(:,2),room32FingersPre(:,48),'o','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room20FingersPre(:,2),room20FingersPre(:,48),'x','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room18FingersPre(:,2),room18FingersPre(:,48),'.','MarkerSize',20,'MarkerFaceColor','red','MarkerEdgeColor','red')
%means:
plot(room6FingersPre(:,2), mean(room6FingersPre(:,48))*ones(length(room6FingersPre(:,2)),1),'LineWidth',2,'Color','black')
plot(room16FingersPre(:,2), mean(room16FingersPre(:,48))*ones(length(room16FingersPre(:,2)),1),'LineWidth',2,'Color','black')
plot(room32FingersPre(:,2), mean(room32FingersPre(:,48))*ones(length(room32FingersPre(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room20FingersPre(:,2), mean(room20FingersPre(:,48))*ones(length(room20FingersPre(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room18FingersPre(:,2), mean(room18FingersPre(:,48))*ones(length(room18FingersPre(:,2)),1),'--','LineWidth',2,'Color','black')
hold off
xlabel('Student Number')
ylabel('Finger Pre-Test Points')
legend('Room 16','Room 6','Room 32','Room 20','Room 18')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0 125])
ylim([0 60])
subplot(2,1,2)
hold on
% computers
plot(room16FingersPost(:,2),room16FingersPost(:,48),'o','MarkerSize',10,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room6FingersPost(:,2),room6FingersPost(:,48),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
%hapcaps
plot(room32FingersPost(:,2),room32FingersPost(:,48),'o','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room20FingersPost(:,2),room20FingersPost(:,48),'x','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room18FingersPost(:,2),room18FingersPost(:,48),'.','MarkerSize',20,'MarkerFaceColor','red','MarkerEdgeColor','red')
%means:
plot(room6FingersPost(:,2), mean(room6FingersPost(:,48))*ones(length(room6FingersPost(:,2)),1),'LineWidth',2,'Color','black')
plot(room16FingersPost(:,2), mean(room16FingersPost(:,48))*ones(length(room16FingersPost(:,2)),1),'LineWidth',2,'Color','black')
plot(room32FingersPost(:,2), mean(room32FingersPost(:,48))*ones(length(room32FingersPost(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room20FingersPost(:,2), mean(room20FingersPost(:,48))*ones(length(room20FingersPost(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room18FingersPost(:,2), mean(room18FingersPost(:,48))*ones(length(room18FingersPost(:,2)),1),'--','LineWidth',2,'Color','black')
hold off
xlabel('Student Number')
ylabel('Finger Post-Test Points')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0 125])
ylim([0 60])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Room32FingersImprovement = room32FingersPost1(:,48)-room32FingersPre1(:,48);
Room32MeanFingersImprovement = mean(Room32FingersImprovement);
Room32StdFingersImprovement = std(Room32FingersImprovement);
Room20FingersImprovement = room20FingersPost1(:,48)-room20FingersPre1(:,48);
Room20MeanFingersImprovement = mean(Room20FingersImprovement);
Room20StdFingersImprovement = std(Room20FingersImprovement);
Room18FingersImprovement = room18FingersPost1(:,48)-room18FingersPre1(:,48);
Room18MeanFingersImprovement = mean(Room18FingersImprovement);
Room18StdFingersImprovement = std(Room18FingersImprovement);
Room16FingersImprovement = room16FingersPost1(:,48)-room16FingersPre1(:,48);
Room16MeanFingersImprovement = mean(Room16FingersImprovement);
Room16StdFingersImprovement = std(Room16FingersImprovement);
Room6FingersImprovement = room6FingersPost1(:,48)-room6FingersPre1(:,48);
Room6MeanFingersImprovement = mean(Room6FingersImprovement);
Room6StdFingersImprovement = std(Room6FingersImprovement);

%plots by group of improvement: Fingers
figure
subplot(1,2,1)
set(gcf,'Color','w')
title('Mean and Standard Deviation of Sudents Improvement in Finger')
hold on
errorbar([1;2;3;4;5], [Room6MeanFingersImprovement; Room16MeanFingersImprovement;...
    Room18MeanFingersImprovement; Room20MeanFingersImprovement; Room32MeanFingersImprovement],...
    [Room6StdFingersImprovement; Room16StdFingersImprovement; Room18StdFingersImprovement;...
    Room20StdFingersImprovement; Room32StdFingersImprovement],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents,1), Room6FingersImprovement,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room16NumStudents,1), Room16FingersImprovement,'.','MarkerSize',20,'Color','blue')
plot(3*ones(room18NumStudents,1), Room18FingersImprovement,'.','MarkerSize',20,'Color','red')
plot(4*ones(room20NumStudents,1), Room20FingersImprovement,'.','MarkerSize',20,'Color','red')
plot(5*ones(room32NumStudents,1), Room32FingersImprovement,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Improvement Points')
xlim([0.6 5.4])
xticks([1.0 2.0 3.0 4.0 5.0])
xticklabels({'Room6', 'Room16', 'Room18', 'Room20', 'Room32'})

HapCapsFingersImprovement = [Room32FingersImprovement; Room20FingersImprovement; Room18FingersImprovement];
HapCapsMeanFingersImprovement = mean(HapCapsFingersImprovement);
HapCapsStdFingersImprovement = std(HapCapsFingersImprovement);
ComputerFingersImprovement = [Room16FingersImprovement; Room6FingersImprovement];
ComputerMeanFingersImprovement = mean(ComputerFingersImprovement);
ComputerStdFingersImprovement = std(ComputerFingersImprovement);

subplot(1,2,2)
set(gcf,'Color','w')
%title('Mean and Standard Deviation of Sudents Improvement by Group in Finger')
hold on
errorbar([1;2], [ComputerMeanFingersImprovement; HapCapsMeanFingersImprovement],...
    [ComputerStdFingersImprovement; HapCapsStdFingersImprovement],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents+room16NumStudents,1), ComputerFingersImprovement,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room18NumStudents + room20NumStudents+room32NumStudents,1), HapCapsFingersImprovement,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Improvement Points')
xlim([0.6 2.4])
xticks([1.0 2.0])
xticklabels({'Computer', 'HapCaps'})

[hFinger,pFinger] = ttest2(HapCapsFingersImprovement,ComputerFingersImprovement,'Vartype','unequal')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Math Analysis%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot all rooms, pre-test and post test: math %39 is max pts.
figure
subplot(2,1,1)
hold on
% computers
plot(room16MathPre(:,2),room16MathPre(:,31),'o','MarkerSize',10,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room6MathPre(:,2),room6MathPre(:,31),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
%hapcaps
plot(room32MathPre(:,2),room32MathPre(:,31),'o','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room20MathPre(:,2),room20MathPre(:,31),'x','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room18MathPre(:,2),room18MathPre(:,31),'.','MarkerSize',20,'MarkerFaceColor','red','MarkerEdgeColor','red')
%means:
plot(room6MathPre(:,2), mean(room6MathPre(:,31))*ones(length(room6MathPre(:,2)),1),'LineWidth',2,'Color','black')
plot(room16MathPre(:,2), mean(room16MathPre(:,31))*ones(length(room16MathPre(:,2)),1),'LineWidth',2,'Color','black')
plot(room32MathPre(:,2), mean(room32MathPre(:,31))*ones(length(room32MathPre(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room20MathPre(:,2), mean(room20MathPre(:,31))*ones(length(room20MathPre(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room18MathPre(:,2), mean(room18MathPre(:,31))*ones(length(room18MathPre(:,2)),1),'--','LineWidth',2,'Color','black')
hold off
xlabel('Student Number')
ylabel('Math Pre-Test Points')
legend('Room 16','Room 6','Room 32','Room 20','Room 18')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0 125])
ylim([0 40])
subplot(2,1,2)
hold on
% computers
plot(room16MathPost1(:,2),room16MathPost1(:,31),'o','MarkerSize',10,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room6MathPost1(:,2),room6MathPost1(:,31),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
%hapcaps
plot(room32MathPost1(:,2),room32MathPost1(:,31),'o','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room20MathPost1(:,2),room20MathPost1(:,31),'x','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room18MathPost1(:,2),room18MathPost1(:,31),'.','MarkerSize',20,'MarkerFaceColor','red','MarkerEdgeColor','red')
%means:
plot(room6MathPost1(:,2), mean(room6MathPost1(:,31))*ones(length(room6MathPost1(:,2)),1),'LineWidth',2,'Color','black')
plot(room16MathPost1(:,2), mean(room16MathPost1(:,31))*ones(length(room16MathPost1(:,2)),1),'LineWidth',2,'Color','black')
plot(room32MathPost1(:,2), mean(room32MathPost1(:,31))*ones(length(room32MathPost1(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room20MathPost1(:,2), mean(room20MathPost1(:,31))*ones(length(room20MathPost1(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room18MathPost1(:,2), mean(room18MathPost1(:,31))*ones(length(room18MathPost1(:,2)),1),'--','LineWidth',2,'Color','black')
hold off
xlabel('Student Number')
ylabel('Math Post-Test Points')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0 125])
ylim([0 40])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Math Analysis%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Room32MathImprovement = room32MathPost1(:,end)-room32MathPre(:,end);
Room32MeanMathImprovement = mean(Room32MathImprovement);
Room32StdMathImprovement = std(Room32MathImprovement);
Room20MathImprovement = room20MathPost1(:,end)-room20MathPre(:,end);
Room20MeanMathImprovement = mean(Room20MathImprovement);
Room20StdMathImprovement = std(Room20MathImprovement);
Room18MathImprovement = room18MathPost1(:,end)-room18MathPre(:,end);
Room18MeanMathImprovement = mean(Room18MathImprovement);
Room18StdMathImprovement = std(Room18MathImprovement);
Room16MathImprovement = room16MathPost1(:,end)-room16MathPre(:,end);
Room16MeanMathImprovement = mean(Room16MathImprovement);
Room16StdMathImprovement = std(Room16MathImprovement);
Room6MathImprovement = room6MathPost1(:,end)-room6MathPre(:,end);
Room6MeanMathImprovement = mean(Room6MathImprovement);
Room6StdMathImprovement = std(Room6MathImprovement);

figure
subplot(1,2,1)
set(gcf,'Color','w')
title('Mean and Standard Deviation of Sudents Improvement in Math')
hold on
errorbar([1;2;3;4;5], [Room6MeanMathImprovement; Room16MeanMathImprovement;...
    Room18MeanMathImprovement; Room20MeanMathImprovement; Room32MeanMathImprovement],...
    [Room6StdMathImprovement; Room16StdMathImprovement; Room18StdMathImprovement;...
    Room20StdMathImprovement; Room32StdMathImprovement],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents,1), Room6MathImprovement,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room16NumStudents,1), Room16MathImprovement,'.','MarkerSize',20,'Color','blue')
plot(3*ones(room18NumStudents,1), Room18MathImprovement,'.','MarkerSize',20,'Color','red')
plot(4*ones(room20NumStudents,1), Room20MathImprovement,'.','MarkerSize',20,'Color','red')
plot(5*ones(room32NumStudents,1), Room32MathImprovement,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Improvement Points')
xlim([0.6 5.4])
xticks([1.0 2.0 3.0 4.0 5.0])
xticklabels({'Room6', 'Room16', 'Room18', 'Room20', 'Room32'})


HapCapsMathImprovement = [Room32MathImprovement; Room20MathImprovement; Room18MathImprovement];
HapCapsMeanMathImprovement = mean(HapCapsMathImprovement);
HapCapsStdMathImprovement = std(HapCapsMathImprovement);
ComputerMathImprovement = [Room16MathImprovement; Room6MathImprovement];
ComputerMeanMathImprovement = mean(ComputerMathImprovement);
ComputerStdMathImprovement = std(ComputerMathImprovement);

subplot(1,2,2)
set(gcf,'Color','w')
%title('Mean and Standard Deviation of Sudents Improvement by Group in Math')
hold on
errorbar([1;2], [ComputerMeanMathImprovement; HapCapsMeanMathImprovement],...
    [ComputerStdMathImprovement; HapCapsStdMathImprovement],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents+room16NumStudents,1), ComputerMathImprovement,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room18NumStudents + room20NumStudents+room32NumStudents,1), HapCapsMathImprovement,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Improvement Points')
xlim([0.6 2.4])
xticks([1.0 2.0])
xticklabels({'Computer', 'HapCaps'})




[hMath,pMath] = ttest2(HapCapsMathImprovement,ComputerMathImprovement,'Vartype','unequal')


%regression analysis
%first by HapCaps vs Computer:

Group = [zeros(length(ComputerMathImprovement),1); ones(length(HapCapsMathImprovement),1)];
Improvement = [ComputerMathImprovement; HapCapsMathImprovement];

b1 = Group\Improvement;

yCalc1 = b1*Group;

figure
scatter(Group,Improvement)
hold on
plot(Group,yCalc1)
hold off
xlabel('Group')
ylabel('improvement points')
title('linear regression relation between group and improvement in math')
grid on
%calculate R2:
Rsq1 = 1 - sum((Improvement - yCalc1).^2)/sum((Improvement - mean(Improvement)).^2);

genderHapCaps = [genderRoom18;genderRoom20;genderRoom32];
genderComputer = [genderRoom6;genderRoom16];


%%% BKT
%Learn Transition Hapcaps

MeanTransitionBKT_HapCaps = 0.9052840265409683;

VarianceTransitionBKT_HapCaps = 0.005056714496268578;

%Learn Transition Computer

MeanTransitionBKT_Computer = 0.9046329057456984;

VarianceTransitionBKT_Computer = 0.0068137908403066785;

%Avg Guess per Question

AvgGuesses_HapCap = [0.72522582; 0.60786242; 0.47273889; 0.75036638;...
    0.66515549; 0.53484804; 0.38536503; 0.424204;   0.62460019;...
    0.58951826; 0.68330475; 0.75278817; 0.64824595; 0.5027727;...
    0.80716903; 0.38218361; 0.61044862; 0.34107215; 0.55602609;...
    0.64408618; 0.64728886; 0.37221206; 0.39753219];

meanGuesses_HapCap = mean(AvgGuesses_HapCap);
stdGuesses_HapCap = std(AvgGuesses_HapCap);

AvgGuesses_Computer = [0.78141319; 0.70642212; 0.57899753; 0.76360613;...
    0.68171775; 0.58790428; 0.53825579; 0.23633649; 0.49046465; 0.70274461;...
    0.66253123; 0.69059426; 0.64875616; 0.68930421; 0.81207501; 0.51177459;...
    0.62336256; 0.40862179; 0.41556987; 0.67283708; 0.70183631; 0.4430947; 0.48290897];

meanGuesses_Computer = mean(AvgGuesses_Computer);
stdGuesses_Computer = std(AvgGuesses_Computer);

%Avg Slip per Question

AvgSlip_HapCap = [0.07779578; 0.12261548; 0.11930765; 0.0605531; 0.11200321;...
    0.14364938; 0.30525045; 0.57471626; 0.08316147; 0.1790163; 0.07414969;...
    0.12794752; 0.02326036; 0.03129882; 0.0857465;  0.30032475; 0.16127622;...
    0.33543798; 0.38827422; 0.16471795; 0.13879829; 0.3805423;  0.47626009];

meanSlip_HapCap = mean(AvgSlip_HapCap);
stdSlip_HapCap = std(AvgSlip_HapCap);

AvgSlip_Computer = [0.05281108; 0.08046016; 0.09363044; 0.04876081; 0.09264934;...
    0.11872019; 0.32502947; 0.53838773; 0.0681894; 0.16664019; 0.05976212; 0.1124457;...
    0.01665728; 0.01497798; 0.07602847; 0.28811854; 0.12131864; 0.24075379;...
    0.34787576; 0.15752019; 0.11324493; 0.30545397; 0.42556853];
meanSlip_Computer = mean(AvgSlip_Computer);
stdSlip_Computer = std(AvgSlip_Computer);

figure
subplot(1,2,1)
set(gcf,'Color','w')
%title('Mean and Standard Deviation of Sudents Improvement by Group in Math')
hold on
errorbar([1;2], [meanGuesses_Computer; meanGuesses_HapCap],...
    [stdGuesses_Computer; stdGuesses_HapCap],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(length(AvgGuesses_Computer),1), AvgGuesses_Computer,'.','MarkerSize',20,'Color','blue')
plot(2*ones(length(AvgGuesses_HapCap),1), AvgGuesses_HapCap,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Average Guesses')
xlim([0.6 2.4])
xticks([1.0 2.0])
xticklabels({'Computer', 'HapCaps'})

subplot(1,2,2)
set(gcf,'Color','w')
%title('Mean and Standard Deviation of Sudents Improvement by Group in Math')
hold on
errorbar([1;2], [meanSlip_Computer; meanSlip_HapCap],...
    [stdSlip_Computer; stdSlip_HapCap],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(length(AvgSlip_Computer),1), AvgSlip_Computer,'.','MarkerSize',20,'Color','blue')
plot(2*ones(length(AvgSlip_HapCap),1), AvgSlip_HapCap,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Average Slip')
xlim([0.6 2.4])
xticks([1.0 2.0])
xticklabels({'Computer', 'HapCaps'})


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Subitizing


%plot all rooms, pre-test and post test: sub fingers % nobody got to max
%poitns, set max to 15
figure
subplot(2,1,1)
hold on
% computers
plot(room16SubFingersPre(:,2),room16SubFingersPre(:,3),'o','MarkerSize',10,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room6SubFingersPre(:,2),room6SubFingersPre(:,3),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
%hapcaps
plot(room32SubFingersPre(:,2),room32SubFingersPre(:,3),'o','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room20SubFingersPre(:,2),room20SubFingersPre(:,3),'x','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room18SubFingersPre(:,2),room18SubFingersPre(:,3),'.','MarkerSize',20,'MarkerFaceColor','red','MarkerEdgeColor','red')
%means:
plot(room6SubFingersPre(:,2), mean(room6SubFingersPre(:,3))*ones(length(room6SubFingersPre(:,2)),1),'LineWidth',2,'Color','black')
plot(room16SubFingersPre(:,2), mean(room16SubFingersPre(:,3))*ones(length(room16SubFingersPre(:,2)),1),'LineWidth',2,'Color','black')
plot(room32SubFingersPre(:,2), mean(room32SubFingersPre(:,3))*ones(length(room32SubFingersPre(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room20SubFingersPre(:,2), mean(room20SubFingersPre(:,3))*ones(length(room20SubFingersPre(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room18SubFingersPre(:,2), mean(room18SubFingersPre(:,3))*ones(length(room18SubFingersPre(:,2)),1),'--','LineWidth',2,'Color','black')
hold off
xlabel('Student Number')
ylabel('Sub-Finger Pre-Test Points')
legend('Room 16','Room 6','Room 32','Room 20','Room 18')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0 125])
ylim([0 15])
subplot(2,1,2)
hold on
% computers
plot(room16SubFingersPost(:,2),room16SubFingersPost(:,3),'o','MarkerSize',10,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room6SubFingersPost(:,2),room6SubFingersPost(:,3),'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
%hapcaps
plot(room32SubFingersPost(:,2),room32SubFingersPost(:,3),'o','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room20SubFingersPost(:,2),room20SubFingersPost(:,3),'x','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room18SubFingersPost(:,2),room18SubFingersPost(:,3),'.','MarkerSize',20,'MarkerFaceColor','red','MarkerEdgeColor','red')
%means:
plot(room6SubFingersPost(:,2), mean(room6SubFingersPost(:,3))*ones(length(room6SubFingersPost(:,2)),1),'LineWidth',2,'Color','black')
plot(room16SubFingersPost(:,2), mean(room16SubFingersPost(:,3))*ones(length(room16SubFingersPost(:,2)),1),'LineWidth',2,'Color','black')
plot(room32SubFingersPost(:,2), mean(room32SubFingersPost(:,3))*ones(length(room32SubFingersPost(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room20SubFingersPost(:,2), mean(room20SubFingersPost(:,3))*ones(length(room20SubFingersPost(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room18SubFingersPost(:,2), mean(room18SubFingersPost(:,3))*ones(length(room18SubFingersPost(:,2)),1),'--','LineWidth',2,'Color','black')
hold off
xlabel('Student Number')
ylabel('Sub-Finger Post-Test Points')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0 125])
ylim([0 15])




Room32SubitizingImprovement = room32SubFingersPost(:,3)-room32SubFingersPre(:,3);
Room32MeanSubitizingImprovement = mean(Room32SubitizingImprovement);
Room32StdSubitizingImprovement = std(Room32SubitizingImprovement);
Room20SubitizingImprovement = room20SubFingersPost(:,3)-room20SubFingersPre(:,3);
Room20MeanSubitizingImprovement = mean(Room20SubitizingImprovement);
Room20StdSubitizingImprovement = std(Room20SubitizingImprovement);
Room18SubitizingImprovement = room18SubFingersPost(:,3)-room18SubFingersPre(:,3);
Room18MeanSubitizingImprovement = mean(Room18SubitizingImprovement);
Room18StdSubitizingImprovement = std(Room18SubitizingImprovement);
Room16SubitizingImprovement = room16SubFingersPost(:,3)-room16SubFingersPre(:,3);
Room16MeanSubitizingImprovement = mean(Room16SubitizingImprovement);
Room16StdSubitizingImprovement = std(Room16SubitizingImprovement);
Room6SubitizingImprovement = room6SubFingersPost(:,3)-room6SubFingersPre(:,3);
Room6MeanSubitizingImprovement = mean(Room6SubitizingImprovement);
Room6StdSubitizingImprovement = std(Room6SubitizingImprovement);


%plots by group of improvement:
figure
subplot(1,2,1)
set(gcf,'Color','w')
title('Mean and Standard Deviation of Sudents Improvement in Subitizing Pictures of Fingers')
hold on
errorbar([1;2;3;4;5], [Room6MeanSubitizingImprovement; Room16MeanSubitizingImprovement;...
    Room18MeanSubitizingImprovement; Room20MeanSubitizingImprovement; Room32MeanSubitizingImprovement],...
    [Room6StdSubitizingImprovement; Room16StdSubitizingImprovement; Room18StdSubitizingImprovement;...
    Room20StdSubitizingImprovement; Room32StdSubitizingImprovement],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents,1), Room6SubitizingImprovement,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room16NumStudents,1), Room16SubitizingImprovement,'.','MarkerSize',20,'Color','blue')
plot(3*ones(room18NumStudents,1), Room18SubitizingImprovement,'.','MarkerSize',20,'Color','red')
plot(4*ones(room20NumStudents,1), Room20SubitizingImprovement,'.','MarkerSize',20,'Color','red')
plot(5*ones(room32NumStudents,1), Room32SubitizingImprovement,'.','MarkerSize',20,'Color','red')
hold off
xlabel('Group Number')
ylabel('Improvement Points')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0.6 5.4])
xticks([1.0 2.0 3.0 4.0 5.0])
xticklabels({'Room6', 'Room16', 'Room18', 'Room20', 'Room32'})

%%%% now by HapCaps and Computer

HapCapsSubitizingImprovement = [Room32SubitizingImprovement; Room20SubitizingImprovement; Room18SubitizingImprovement];
HapCapsMeanSubitizingImprovement = mean(HapCapsSubitizingImprovement);
HapCapsStdSubitizingImprovement = std(HapCapsSubitizingImprovement);
ComputerSubitizingImprovement = [Room16SubitizingImprovement; Room6SubitizingImprovement];
ComputerMeanSubitizingImprovement = mean(ComputerSubitizingImprovement);
ComputerStdSubitizingImprovement = std(ComputerSubitizingImprovement);

subplot(1,2,2)
set(gcf,'Color','w')
%title('Mean and Standard Deviation of Sudents Improvement by Group in Subitizing')
hold on
errorbar([1;2], [ComputerMeanSubitizingImprovement; HapCapsMeanSubitizingImprovement],...
    [ComputerStdSubitizingImprovement; HapCapsStdSubitizingImprovement],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents+room16NumStudents,1), ComputerSubitizingImprovement,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room18NumStudents + room20NumStudents+room32NumStudents,1), HapCapsSubitizingImprovement,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Improvement Points')
xlim([0.6 2.4])
xticks([1.0 2.0])
xticklabels({'Computer', 'HapCaps'})

[hPSubitizing,pSubitizing] = ttest2(HapCapsSubitizingImprovement,ComputerSubitizingImprovement,'Vartype','unequal')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Puzzles Analysis%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
room16NumFour = room16Puzzles(:,3);
room16PuzzlesTotals = room16Puzzles(:,22)-room16NumFour;
room6NumFour = room6Puzzles(:,3);
room6PuzzlesTotals = room6Puzzles(:,22)-room6NumFour;
room18NumFour = room18Puzzles(:,3);
room18PuzzlesTotals = room18Puzzles(:,22)-room18NumFour;
room20NumFour = room20Puzzles(:,3);
room20PuzzlesTotals = room20Puzzles(:,22)-room20NumFour;
room32NumFour = room32Puzzles(:,3);
room32PuzzlesTotals = room32Puzzles(:,22)-room32NumFour;


%plot all rooms puzzle test: math %22 is max pts.
figure
subplot(2,1,1)
hold on
% computers
plot(room16Puzzles(:,2),room16PuzzlesTotals,'o','MarkerSize',10,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room6Puzzles(:,2),room6PuzzlesTotals,'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
%hapcaps
plot(room32Puzzles(:,2),room32PuzzlesTotals,'o','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room20MathPre(:,2),room20PuzzlesTotals,'x','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room18MathPre(:,2),room18PuzzlesTotals,'.','MarkerSize',20,'MarkerFaceColor','red','MarkerEdgeColor','red')
%means:
plot(room6Puzzles(:,2), mean(room6PuzzlesTotals)*ones(length(room6Puzzles(:,2)),1),'LineWidth',2,'Color','black')
plot(room16Puzzles(:,2), mean(room16PuzzlesTotals)*ones(length(room16Puzzles(:,2)),1),'LineWidth',2,'Color','black')
plot(room32Puzzles(:,2), mean(room32PuzzlesTotals)*ones(length(room32Puzzles(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room20Puzzles(:,2), mean(room20PuzzlesTotals)*ones(length(room20Puzzles(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room18Puzzles(:,2), mean(room18PuzzlesTotals)*ones(length(room18Puzzles(:,2)),1),'--','LineWidth',2,'Color','black')
hold off
xlabel('Student Number')
ylabel('Math Transference Points')
legend('Room 16','Room 6','Room 32','Room 20','Room 18')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0 125])
ylim([0 22])

%now plotting how much they saw the number 4
subplot(2,1,2)
hold on
% computers
plot(room16Puzzles(:,2),room16NumFour,'o','MarkerSize',10,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
plot(room6Puzzles(:,2),room6NumFour,'.','MarkerSize',20,'MarkerFaceColor','blue','MarkerEdgeColor','blue')
%hapcaps
plot(room32Puzzles(:,2),room32NumFour,'o','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room20Puzzles(:,2),room20NumFour,'x','MarkerSize',10,'MarkerFaceColor','red','MarkerEdgeColor','red')
plot(room18Puzzles(:,2),room18NumFour,'.','MarkerSize',20,'MarkerFaceColor','red','MarkerEdgeColor','red')
%means:
plot(room6Puzzles(:,2), mean(room6NumFour)*ones(length(room6Puzzles(:,2)),1),'LineWidth',2,'Color','black')
plot(room16Puzzles(:,2), mean(room16NumFour)*ones(length(room16Puzzles(:,2)),1),'LineWidth',2,'Color','black')
plot(room32Puzzles(:,2), mean(room32NumFour)*ones(length(room32Puzzles(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room20Puzzles(:,2), mean(room20NumFour)*ones(length(room20Puzzles(:,2)),1),'--','LineWidth',2,'Color','black')
plot(room18Puzzles(:,2), mean(room18NumFour)*ones(length(room18Puzzles(:,2)),1),'--','LineWidth',2,'Color','black')
hold off
xlabel('Student Number')
ylabel('Number Four Points')
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlim([0 125])
ylim([0 15]) %max I think is 10


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Puzzles Analysis%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
room32MeanPuzzles = mean(room32PuzzlesTotals);
room32StdPuzzles = std(room32PuzzlesTotals);
room20MeanPuzzles = mean(room20PuzzlesTotals);
room20StdPuzzles = std(room20PuzzlesTotals);
room18MeanPuzzles = mean(room18PuzzlesTotals);
room18StdPuzzles = std(room18PuzzlesTotals);
room16MeanPuzzles = mean(room16PuzzlesTotals);
room16StdPuzzles = std(room16PuzzlesTotals);
room6MeanPuzzles = mean(room6PuzzlesTotals);
room6StdPuzzles = std(room6PuzzlesTotals);

figure
subplot(1,2,1)
set(gcf,'Color','w')
title('Mean and Standard Deviation of Students Transference Test')
hold on
errorbar([1;2;3;4;5], [room6MeanPuzzles; room16MeanPuzzles;...
    room18MeanPuzzles; room20MeanPuzzles; room32MeanPuzzles],...
    [room6StdPuzzles; room16StdPuzzles; room18StdPuzzles;...
    room20StdPuzzles; room32StdPuzzles],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents,1), room6PuzzlesTotals,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room16NumStudents,1), room16PuzzlesTotals,'.','MarkerSize',20,'Color','blue')
plot(3*ones(room18NumStudents,1), room18PuzzlesTotals,'.','MarkerSize',20,'Color','red')
plot(4*ones(room20NumStudents,1), room20PuzzlesTotals,'.','MarkerSize',20,'Color','red')
plot(5*ones(room32NumStudents,1), room32PuzzlesTotals,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Transference Test Points')
xlim([0.6 5.4])
xticks([1.0 2.0 3.0 4.0 5.0])
xticklabels({'room6', 'room16', 'room18', 'room20', 'room32'})


HapCapsPuzzles = [room32PuzzlesTotals; room20PuzzlesTotals; room18PuzzlesTotals];
HapCapsMeanPuzzles = mean(HapCapsPuzzles);
HapCapsStdPuzzles = std(HapCapsPuzzles);
ComputerPuzzles = [room16PuzzlesTotals; room6PuzzlesTotals];
ComputerMeanPuzzles = mean(ComputerPuzzles);
ComputerStdPuzzles = std(ComputerPuzzles);

subplot(1,2,2)
set(gcf,'Color','w')
%title('Mean and Standard Deviation of Sudents Transference test')
hold on
errorbar([1;2], [ComputerMeanPuzzles; HapCapsMeanPuzzles],...
    [ComputerStdPuzzles; HapCapsStdPuzzles],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents+room16NumStudents,1), ComputerPuzzles,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room18NumStudents + room20NumStudents+room32NumStudents,1), HapCapsPuzzles,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Transference Test Points')
xlim([0.6 2.4])
xticks([1.0 2.0])
xticklabels({'Computer', 'HapCaps'})


[hPuzzles,pPuzzles] = ttest2(HapCapsPuzzles,ComputerPuzzles,'Vartype','unequal')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NumFour Analysis%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

room32MeanNumFour = mean(room32NumFour);
room32StdNumFour = std(room32NumFour);
room20MeanNumFour = mean(room20NumFour);
room20StdNumFour = std(room20NumFour);
room18MeanNumFour = mean(room18NumFour);
room18StdNumFour = std(room18NumFour);
room16MeanNumFour = mean(room16NumFour);
room16StdNumFour = std(room16NumFour);
room6MeanNumFour = mean(room6NumFour);
room6StdNumFour = std(room6NumFour);

figure
subplot(1,2,1)
set(gcf,'Color','w')
title('Mean and Standard Deviation of Sudents Identifying the Number Four')
hold on
errorbar([1;2;3;4;5], [room6MeanNumFour; room16MeanNumFour;...
    room18MeanNumFour; room20MeanNumFour; room32MeanNumFour],...
    [room6StdNumFour; room16StdNumFour; room18StdNumFour;...
    room20StdNumFour; room32StdNumFour],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents,1), room6NumFour,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room16NumStudents,1), room16NumFour,'.','MarkerSize',20,'Color','blue')
plot(3*ones(room18NumStudents,1), room18NumFour,'.','MarkerSize',20,'Color','red')
plot(4*ones(room20NumStudents,1), room20NumFour,'.','MarkerSize',20,'Color','red')
plot(5*ones(room32NumStudents,1), room32NumFour,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('NumFour Test Points')
xlim([0.6 5.4])
ylim([0 10])
xticks([1.0 2.0 3.0 4.0 5.0])
xticklabels({'room6', 'room16', 'room18', 'room20', 'room32'})


HapCapsNumFour = [room32NumFour; room20NumFour; room18NumFour];
HapCapsMeanNumFour = mean(HapCapsNumFour);
HapCapsStdNumFour = std(HapCapsNumFour);
ComputerNumFour = [room16NumFour; room6NumFour];
ComputerMeanNumFour = mean(ComputerNumFour);
ComputerStdNumFour = std(ComputerNumFour);

subplot(1,2,2)
set(gcf,'Color','w')
%title('Mean and Standard Deviation of Identifying the Number Four')
hold on
errorbar([1;2], [ComputerMeanNumFour; HapCapsMeanNumFour],...
    [ComputerStdNumFour; HapCapsStdNumFour],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents+room16NumStudents,1), ComputerNumFour,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room18NumStudents + room20NumStudents+room32NumStudents,1), HapCapsNumFour,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('NumberFour Test Points')
xlim([0.6 2.4])
ylim([0 10])
xticks([1.0 2.0])
xticklabels({'Computer', 'HapCaps'})


[hNumFour,pNumFour] = ttest2(HapCapsNumFour,ComputerNumFour,'Vartype','unequal')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Paper Clip%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

room32MeanPaperClip = mean(room32PaperClip(:,3));
room32StdPaperClip = std(room32PaperClip(:,3));
room20MeanPaperClip = mean(room20PaperClip(:,3));
room20StdPaperClip = std(room20PaperClip(:,3));
room18MeanPaperClip = mean(room18PaperClip(:,3));
room18StdPaperClip = std(room18PaperClip(:,3));
room16MeanPaperClip = mean(room16PaperClip(:,3));
room16StdPaperClip = std(room16PaperClip(:,3));
room6MeanPaperClip = mean(room6PaperClip(:,3));
room6StdPaperClip = std(room6PaperClip(:,3));

figure
subplot(1,2,1)
set(gcf,'Color','w')
title('Mean and Standard Deviation of Sudents Drawing Paper Clip')
hold on
errorbar([1;2;3;4;5], [room6MeanPaperClip; room16MeanPaperClip;...
    room18MeanPaperClip; room20MeanPaperClip; room32MeanPaperClip],...
    [room6StdPaperClip; room16StdPaperClip; room18StdPaperClip;...
    room20StdPaperClip; room32StdPaperClip],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents,1), room6PaperClip(:,3),'.','MarkerSize',20,'Color','blue')
plot(2*ones(room16NumStudents,1), room16PaperClip(:,3),'.','MarkerSize',20,'Color','blue')
plot(3*ones(room18NumStudents,1), room18PaperClip(:,3),'.','MarkerSize',20,'Color','red')
plot(4*ones(room20NumStudents,1), room20PaperClip(:,3),'.','MarkerSize',20,'Color','red')
plot(5*ones(room32NumStudents,1), room32PaperClip(:,3),'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Paper Clip Test Points')
xlim([0.6 5.4])
ylim([0 15])
xticks([1.0 2.0 3.0 4.0 5.0])
xticklabels({'room6', 'room16', 'room18', 'room20', 'room32'})


HapCapsPaperClip = [room32PaperClip(:,3); room20PaperClip(:,3); room18PaperClip(:,3)];
HapCapsMeanPaperClip = mean(HapCapsPaperClip);
HapCapsStdPaperClip = std(HapCapsPaperClip);
ComputerPaperClip = [room16PaperClip(:,3); room6PaperClip(:,3)];
ComputerMeanPaperClip = mean(ComputerPaperClip);
ComputerStdPaperClip = std(ComputerPaperClip);

subplot(1,2,2)
set(gcf,'Color','w')
%title('Mean and Standard Deviation of Identifying the Number Four')
hold on
errorbar([1;2], [ComputerMeanPaperClip; HapCapsMeanPaperClip],...
    [ComputerStdPaperClip; HapCapsStdPaperClip],'x', 'MarkerSize',40,'Color',[0,0,0]);
plot(ones(room6NumStudents+room16NumStudents,1), ComputerPaperClip,'.','MarkerSize',20,'Color','blue')
plot(2*ones(room18NumStudents + room20NumStudents+room32NumStudents,1), HapCapsPaperClip,'.','MarkerSize',20,'Color','red')
hold off
set(gca,'FontSize', 16);
set(gcf,'color','w')
xlabel('Group Number')
ylabel('Paper Clip Test Points')
xlim([0.6 2.4])
ylim([0 15])
xticks([1.0 2.0])
xticklabels({'Computer', 'HapCaps'})


[hPaperClip,pPaperClip] = ttest2(HapCapsPaperClip,ComputerPaperClip,'Vartype','unequal')









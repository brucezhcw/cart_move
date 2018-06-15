// ImitationDlg.cpp : 实现文件
//
#include "stdafx.h"
#include "Imitation.h"
#include "ImitationDlg.h"
#include "afxdialogex.h"
#include <opencv.hpp>
#include <Kinect.h>
#include <thread>
#include <time.h>
#include "Motor.h"
#include <io.h> 
#include <fcntl.h>
#include <conio.h>
#include <vector>
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;
using namespace cv;

#pragma region Global Variables
const float Alpha[7] = { 0, -pi / 2, pi / 2, -pi / 2, -pi / 2, -pi / 2, pi / 2 };
const float Theta[7] = { 0, 0, 0, -pi / 2, -pi / 2, 0, 0 };
bool Run;
double theta[7];
Motor *myMotor;

bool enableMotor = false;
bool openedMotor = false;
thread *MainThread;

int outputPeriod = 2;
int filtersize = 50;
#pragma endregion
#pragma region Global Function

void inverse_kinematics()
{

}

void draw_line(Mat& img, int* angle)
{
	int length_ = 8; 
	int startpointx = 250;
	int startpointy = 250;
	int endpointx = startpointx + int(length_*cos(angle[1] / 180.0 * 3.14159));
	int endpointy = startpointy + int(length_*sin(angle[1] / 180.0 * 3.14159));
	for (int i = 0; i < 24; i ++)
	{
		line(img, Point(startpointx, startpointy), Point(endpointx, endpointy), Scalar(255, 255, 255), -1);
		startpointx = endpointx;
		startpointy = endpointy;
		endpointx = startpointx + int(length_*cos(angle[i + 2] / 180.0 * 3.14159));
		endpointy = startpointy + int(length_*sin(angle[i + 2] / 180.0 * 3.14159));
	}
}


void Tracking() 
{
	Sleep(100);

	Mat show_result(500, 500, CV_32FC1);
	for (int i = 0; i < 500; i++)
		for (int j = 0; j < 500; j++)
			show_result.at<UINT8>(i, j) = 0;

	ifstream in("C:/Users/20600/Desktop/test.txt", ios::in);
	int angle[26];
	for (int i = 0; i < 26; i++)
		in >> angle[i];
	draw_line(show_result, angle);
	cvNamedWindow("cart", 0);
	imshow("processed cart", show_result);
	if (openedMotor && enableMotor)
		{
			double pos[6];
			pos[0] = 500; pos[1] = 0; pos[2] = -200; pos[3] = 0; pos[4] = 0; pos[5] = 0;
			myMotor->Move_J(Left, pos);
			int length_ = 5;
			int endpointx = pos[0];
			int endpointy = pos[1];
			for (int k = 1; k < 25; k++)
			{
			//int lasttime = clock();
			cvWaitKey(1);
			endpointx = endpointx + int(length_*cos(angle[k] / 180.0 * 3.14159));
			endpointy = endpointy + int(length_*sin(angle[k] / 180.0 * 3.14159));
			pos[0] = endpointx;
			pos[1] = endpointy;
			myMotor->Move_J(Left, pos);//Move(Left, theta);
			}
		}
	
}
void InitConsoleWindow()
{
	AllocConsole();
	HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
	int hCrt = _open_osfhandle((long)handle, _O_TEXT);
	FILE * hf = _fdopen(hCrt, "w");
	*stdout = *hf;
}
#pragma endregion
//-----------------------------------------

// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOk();
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
	ON_BN_CLICKED(IDOK, &CAboutDlg::OnBnClickedOk)
END_MESSAGE_MAP()


// CImitationDlg 对话框



CImitationDlg::CImitationDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_IMITATION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CImitationDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CImitationDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CImitationDlg::OnBnClickedOk)
	ON_BN_CLICKED(openKinect, &CImitationDlg::OnBnClickedopenkinect)
	ON_BN_CLICKED(openMotor, &CImitationDlg::OnBnClickedopenmotor)
	ON_BN_CLICKED(releaseKinect, &CImitationDlg::OnBnClickedreleasekinect)
	ON_BN_CLICKED(startTracking, &CImitationDlg::OnBnClickedstarttracking)
	ON_BN_CLICKED(closeMotor, &CImitationDlg::OnBnClickedclosemotor)
	ON_BN_CLICKED(stopTracking, &CImitationDlg::OnBnClickedstoptracking)
	ON_BN_CLICKED(startImitation, &CImitationDlg::OnBnClickedstartimitation)
	ON_BN_CLICKED(stopImitation, &CImitationDlg::OnBnClickedstopimitation)
	ON_BN_CLICKED(motorHome, &CImitationDlg::OnBnClickedmotorhome)
	ON_BN_CLICKED(Photo, &CImitationDlg::OnBnClickedPhoto)
	ON_BN_CLICKED(ID_APP_ABOUT, &CImitationDlg::OnBnClickedAppAbout)
END_MESSAGE_MAP()


// CImitationDlg 消息处理程序

BOOL CImitationDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	InitConsoleWindow();
	_cprintf("Open console OK\n\n");
	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CImitationDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CImitationDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CImitationDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CImitationDlg::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnOK();
}

void CImitationDlg::OnBnClickedopenkinect()
{
	// TODO: 在此添加控件通知处理程序代码
}


void CImitationDlg::OnBnClickedopenmotor()
{
	// TODO: 在此添加控件通知处理程序代码
	myMotor = new Motor(5);
	myMotor->servoOpen();
	openedMotor = true;
	_cprintf("Motor Opend!\n\n");
}


void CImitationDlg::OnBnClickedreleasekinect()
{
	// TODO: 在此添加控件通知处理程序代码
}


void CImitationDlg::OnBnClickedstarttracking()
{
	// TODO: 在此添加控件通知处理程序代码
	Run = true;
	MainThread = new thread(Tracking);
	_cprintf("Is Tracking now....\n\n");
}


void CImitationDlg::OnBnClickedclosemotor()
{
	// TODO: 在此添加控件通知处理程序代码
	myMotor->servoClose();
	openedMotor = false;
	_cprintf("Motor is closed!\n\n");
}


void CImitationDlg::OnBnClickedstoptracking()
{
	// TODO: 在此添加控件通知处理程序代码
	Run = false;
	_cprintf("Tracking is stopped!\n");
}


void CImitationDlg::OnBnClickedstartimitation()
{
	// TODO: 在此添加控件通知处理程序代码
	Sleep(5000);
	enableMotor = true;
	_cprintf("Motor is imitating.....\n\n");
}


void CImitationDlg::OnBnClickedstopimitation()
{
	// TODO: 在此添加控件通知处理程序代码
	enableMotor = false;
	_cprintf("Motor is not imitating any more\n\n");
}


void CImitationDlg::OnBnClickedmotorhome()
{
	// TODO: 在此添加控件通知处理程序代码
	_cprintf("Motor is going to Home position....");
	myMotor->Home(Left);
	_cprintf("Motor is Home now\n\n");
}


void CImitationDlg::OnBnClickedPhoto()
{
	// TODO: 在此添加控件通知处理程序代码
}


void CImitationDlg::OnBnClickedAppAbout()
{
	// TODO:  在此添加控件通知处理程序代码
}


void CAboutDlg::OnBnClickedOk()
{
	// TODO:  在此添加控件通知处理程序代码
	CDialogEx::OnOK();
}
